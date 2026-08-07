// Copyright 2026 The Magma GPU Project
// SPDX-License-Identifier: MIT

use std::sync::atomic::Ordering;
use std::sync::Arc;

use crate::magma::device_state::MagmaDeviceState;
use crate::rutabaga_utils::RutabagaError;
use crate::rutabaga_utils::RutabagaResult;
use magma_gpu::util::Reader;
use magma_gpu_magma::protocol::*;

pub struct MagmaDispatcher {
    state: Arc<MagmaDeviceState>,
    default_queue_id: u32,
}

impl MagmaDispatcher {
    pub fn new(state: Arc<MagmaDeviceState>, default_queue_id: u32) -> Self {
        Self {
            state,
            default_queue_id,
        }
    }

    pub fn dispatch(&self, msg: MagmaProtocol, reader: &mut Reader) -> RutabagaResult<()> {
        eprintln!("MagmaDispatcher::dispatch: {msg:?}");
        match msg {
            MagmaProtocol::CreateDevice(cmd) => {
                let pdev_idx = cmd.physical_device as usize;
                if pdev_idx >= self.state.physical_devices.len() {
                    return Err(RutabagaError::InvalidResourceId);
                }
                let pdev = &self.state.physical_devices[pdev_idx];
                let dev = pdev
                    .create_device()
                    .map_err(|_| RutabagaError::InvalidResourceId)?;
                self.state.set_device(dev);
                Ok(())
            }
            MagmaProtocol::CreateAddressSpace(_cmd) => {
                let dev = self
                    .state
                    .get_device()
                    .ok_or(RutabagaError::InvalidResourceId)?;
                let addr_space = dev
                    .create_address_space()
                    .map_err(|_| RutabagaError::InvalidResourceId)?;
                let id = self.state.object_id.fetch_add(1, Ordering::Relaxed);
                self.state
                    .address_spaces
                    .write()
                    .unwrap()
                    .insert(id, Arc::new(addr_space));
                Ok(())
            }
            MagmaProtocol::CreateQueue(cmd) => {
                let dev = self
                    .state
                    .get_device()
                    .ok_or(RutabagaError::InvalidResourceId)?;
                let as_map = self.state.address_spaces.read().unwrap();
                let addr_space = as_map
                    .get(&cmd.address_space)
                    .ok_or(RutabagaError::InvalidResourceId)?;
                let queue = dev
                    .create_queue(addr_space, &cmd.info)
                    .map_err(|_| RutabagaError::InvalidResourceId)?;
                let id = self.state.object_id.fetch_add(1, Ordering::Relaxed);
                self.state
                    .queues
                    .write()
                    .unwrap()
                    .insert(id, Arc::new(queue));
                Ok(())
            }
            MagmaProtocol::CreateBuffer(_cmd) => {
                let dev = self
                    .state
                    .get_device()
                    .ok_or(RutabagaError::InvalidResourceId)?;
                let info: MagmaCreateBufferInfo = reader
                    .read_try_obj()
                    .map_err(|_| RutabagaError::InvalidCommandBuffer)?;
                let buffer = dev
                    .create_buffer(&info)
                    .map_err(|_| RutabagaError::InvalidResourceId)?;
                if let Ok(mapped) = buffer.clone().map() {
                    unsafe {
                        let ptr = mapped.as_ptr() as *mut u32;
                        *ptr = 0xffff1000;
                    }
                }
                let id = self.state.object_id.fetch_add(1, Ordering::Relaxed);
                self.state
                    .buffers
                    .write()
                    .unwrap()
                    .insert(id, Arc::new(buffer));
                Ok(())
            }
            MagmaProtocol::MapBufferGpu(cmd) => {
                let as_map = self.state.address_spaces.read().unwrap();
                let addr_space = as_map
                    .get(&cmd.address_space)
                    .ok_or(RutabagaError::InvalidResourceId)?;
                let buf_map = self.state.buffers.read().unwrap();
                let buffer = buf_map
                    .get(&cmd.buffer)
                    .ok_or(RutabagaError::InvalidResourceId)?;
                addr_space
                    .map_buffer_gpu(buffer, cmd.buffer_offset, cmd.gpu_va, cmd.size, cmd.flags)
                    .map_err(|_| RutabagaError::InvalidResourceId)?;
                Ok(())
            }
            MagmaProtocol::UnmapBufferGpu(cmd) => {
                let as_map = self.state.address_spaces.read().unwrap();
                let addr_space = as_map
                    .get(&cmd.address_space)
                    .ok_or(RutabagaError::InvalidResourceId)?;
                addr_space
                    .unmap_buffer_gpu(cmd.gpu_va, cmd.size)
                    .map_err(|_| RutabagaError::InvalidResourceId)?;
                Ok(())
            }
            MagmaProtocol::SubmitCommand(cmd) => {
                // TODO: Formalize wait_seqno in Gorgonzola instead of using _pad0.
                if cmd._padding > 0 {
                    self.state.wait_for_cpu_seqno(cmd._padding as u64);
                }

                let queue_id = if cmd.queue != 0 {
                    cmd.queue
                } else {
                    self.default_queue_id
                };
                let queues = self.state.queues.read().unwrap();
                let queue = match queues.get(&queue_id) {
                    Some(q) => q,
                    None => {
                        eprintln!(
                            "SubmitCommand: queue_id {queue_id} not found! available queues: {:?}",
                            queues.keys().collect::<Vec<_>>()
                        );
                        return Err(RutabagaError::InvalidResourceId);
                    }
                };

                let mut submit_info: MagmaSubmitInfo = reader
                    .read_try_obj()
                    .map_err(|_| RutabagaError::InvalidCommandBuffer)?;
                let mut next_ptr = submit_info.header.p_next;
                submit_info.header.p_next = 0;

                let mut as_info_storage = None;
                let mut buf_info_storage = None;

                while next_ptr != 0 {
                    let hdr = reader
                        .peek_obj::<MagmaStructureTypeHeader>()
                        .map_err(|_| RutabagaError::InvalidCommandBuffer)?;
                    match hdr.stype {
                        MAGMA_STRUCTURE_TYPE_SUBMIT_ADDRESS_SPACE_INFO => {
                            let mut ext: MagmaSubmitAddressSpaceInfo = reader
                                .read_try_obj()
                                .map_err(|_| RutabagaError::InvalidCommandBuffer)?;
                            next_ptr = ext.header.p_next;
                            ext.header.p_next = 0;
                            as_info_storage = Some(ext);
                        }
                        MAGMA_STRUCTURE_TYPE_SUBMIT_BUFFER_INFO => {
                            let mut ext: MagmaSubmitBufferInfo = reader
                                .read_try_obj()
                                .map_err(|_| RutabagaError::InvalidCommandBuffer)?;
                            next_ptr = ext.header.p_next;
                            ext.header.p_next = 0;
                            buf_info_storage = Some(ext);
                        }
                        _ => return Err(RutabagaError::InvalidCommandBuffer),
                    }
                }

                if let Some(ext) = as_info_storage.as_mut() {
                    unsafe { submit_info.push_next(ext) };
                }
                if let Some(ext) = buf_info_storage.as_mut() {
                    unsafe { submit_info.push_next(ext) };
                }

                if let Some(sync_info) = submit_info.sync_info() {
                    let sync_objs = self.state.sync_objs.read().unwrap();
                    let mut translated = *sync_info;
                    for i in 0..translated.num_wait_sync_objs as usize {
                        let guest_id = translated.wait_sync_objs[i];
                        if let Some(so) = sync_objs.get(&guest_id) {
                            translated.wait_sync_objs[i] = so.as_raw_handle().unwrap_or(0);
                        }
                    }
                    for i in 0..translated.num_signal_sync_objs as usize {
                        let guest_id = translated.signal_sync_objs[i];
                        if let Some(so) = sync_objs.get(&guest_id) {
                            translated.signal_sync_objs[i] = so.as_raw_handle().unwrap_or(0);
                        }
                    }
                    submit_info.set_sync_info(translated);
                }
                queue.submit_command(&submit_info).map_err(|e| {
                    eprintln!("queue.submit_command failed: {e:?}");
                    RutabagaError::InvalidCommandBuffer
                })?;

                Ok(())
            }
            MagmaProtocol::CreateSyncObj(_cmd) => {
                let dev = self
                    .state
                    .get_device()
                    .ok_or(RutabagaError::InvalidResourceId)?;
                let info: MagmaCreateSyncObjInfo = reader
                    .read_try_obj()
                    .map_err(|_| RutabagaError::InvalidCommandBuffer)?;
                let sync_obj = dev
                    .create_sync_obj(&info)
                    .map_err(|_| RutabagaError::InvalidResourceId)?;
                let id = self.state.object_id.fetch_add(1, Ordering::Relaxed);
                self.state
                    .sync_objs
                    .write()
                    .unwrap()
                    .insert(id, Arc::new(sync_obj));
                Ok(())
            }
            MagmaProtocol::SyncObjWait(cmd) => {
                let sync_objs = self.state.sync_objs.read().unwrap();
                let sync_obj = sync_objs
                    .get(&cmd.sync_obj)
                    .ok_or(RutabagaError::InvalidResourceId)?;
                sync_obj
                    .wait(cmd.timeout_ns)
                    .map_err(|_| RutabagaError::InvalidResourceId)?;
                Ok(())
            }
            MagmaProtocol::SyncObjSignal(cmd) => {
                let sync_objs = self.state.sync_objs.read().unwrap();
                let sync_obj = sync_objs
                    .get(&cmd.sync_obj)
                    .ok_or(RutabagaError::InvalidResourceId)?;
                sync_obj
                    .signal()
                    .map_err(|_| RutabagaError::InvalidResourceId)?;
                Ok(())
            }
            MagmaProtocol::SyncObjClose(_cmd) => Ok(()),
            _ => Ok(()),
        }
    }
}
