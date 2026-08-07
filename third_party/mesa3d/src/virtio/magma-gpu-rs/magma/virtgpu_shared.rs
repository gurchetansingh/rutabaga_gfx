// Copyright 2026 The Magma GPU Project
// SPDX-License-Identifier: MIT

#![allow(dead_code)]

use magma_gpu::util::Error as MagmaGpuError;
use magma_gpu::util::Reader;
use magma_gpu::util::Result as MagmaGpuResult;

use crate::decoder::decode_extensible_struct;
use crate::encoder::Encoder;
use crate::magma_defines::MagmaCreateBufferInfo;
use crate::magma_defines::MagmaCreateQueueInfo;
use crate::magma_defines::MagmaHeap;
use crate::magma_defines::MagmaHeapBudget;
use crate::magma_defines::MagmaMemoryType;
use crate::magma_defines::MagmaPhysicalDeviceInfo;
use crate::protocol::*;

#[derive(Default, Clone, Debug)]
pub struct DecodedCapabilities {
    pub caps: MagmaVirtCapabilities,
    pub devices: Vec<DecodedPhysicalDevice>,
}

#[derive(Default, Clone, Debug)]
pub struct DecodedPhysicalDevice {
    pub info: MagmaPhysicalDeviceInfo,
    pub memory_types: Vec<MagmaMemoryType>,
    pub memory_heaps: Vec<MagmaHeap>,
}

pub fn decode_capabilities(buf: &[u8]) -> MagmaGpuResult<DecodedCapabilities> {
    let mut reader = Reader::new(buf);
    let mut decoded = DecodedCapabilities::default();
    let mut current_device: Option<DecodedPhysicalDevice> = None;

    while let Some(ext) = decode_extensible_struct(&mut reader) {
        match ext {
            MagmaExtensibleStruct::VirtCapabilities(caps) => {
                decoded.caps = caps;
            }
            MagmaExtensibleStruct::PhysicalDeviceInfo(info) => {
                if let Some(dev) = current_device.take() {
                    decoded.devices.push(dev);
                }
                current_device = Some(DecodedPhysicalDevice {
                    info,
                    ..Default::default()
                });
            }
            MagmaExtensibleStruct::PhysicalDeviceMemoryType(mt) => {
                if let Some(ref mut dev) = current_device {
                    dev.memory_types.push(MagmaMemoryType {
                        property_flags: mt.property_flags,
                        heap_idx: mt.heap_idx,
                    });
                }
            }
            MagmaExtensibleStruct::PhysicalDeviceMemoryHeap(mh) => {
                if let Some(ref mut dev) = current_device {
                    dev.memory_heaps.push(MagmaHeap {
                        heap_size: mh.heap_size,
                        heap_flags: mh.heap_flags,
                    });
                }
            }
            _ => {}
        }
    }

    if let Some(dev) = current_device {
        decoded.devices.push(dev);
    }

    Ok(decoded)
}

pub fn encode_and_submit_create_device<F>(submit: F) -> MagmaGpuResult<()>
where
    F: FnOnce(&[u8]) -> MagmaGpuResult<()>,
{
    let mut buf = [0u8; 256];
    let mut encoder = Encoder::new(&mut buf);
    let req = CreateDevice {
        header: MagmaCommandHeader {
            opcode: MAGMA_OPCODE_CREATE_DEVICE,
            size: std::mem::size_of::<CreateDevice>() as u32,
        },
        ..Default::default()
    };
    encoder.encode_create_device(&req)?;
    let len = encoder.bytes_written();
    submit(&buf[..len])
}

pub fn encode_and_submit_get_memory_budget<F>(
    heap_idx: u32,
    submit: F,
) -> MagmaGpuResult<MagmaHeapBudget>
where
    F: FnOnce(&[u8]) -> MagmaGpuResult<()>,
{
    let mut buf = [0u8; 256];
    let mut encoder = Encoder::new(&mut buf);
    let req = GetMemoryBudget {
        header: MagmaCommandHeader {
            opcode: MAGMA_OPCODE_GET_MEMORY_BUDGET,
            size: std::mem::size_of::<GetMemoryBudget>() as u32,
        },
        device: 0,
        heap_idx,
    };
    encoder.encode_get_memory_budget(&req)?;
    let len = encoder.bytes_written();
    submit(&buf[..len])?;
    Err(MagmaGpuError::Unsupported)
}

pub fn encode_and_submit_create_address_space<F>(submit: F) -> MagmaGpuResult<()>
where
    F: FnOnce(&[u8]) -> MagmaGpuResult<()>,
{
    let mut buf = [0u8; 256];
    let mut encoder = Encoder::new(&mut buf);
    let req = CreateAddressSpace {
        header: MagmaCommandHeader {
            opcode: MAGMA_OPCODE_CREATE_ADDRESS_SPACE,
            size: std::mem::size_of::<CreateAddressSpace>() as u32,
        },
        device: 0,
        ..Default::default()
    };
    encoder.encode_create_address_space(&req)?;
    let len = encoder.bytes_written();
    submit(&buf[..len])
}

pub fn encode_and_submit_create_queue<F>(
    address_space: u32,
    info: &MagmaCreateQueueInfo,
    submit: F,
) -> MagmaGpuResult<()>
where
    F: FnOnce(&[u8]) -> MagmaGpuResult<()>,
{
    let mut buf = [0u8; 256];
    let mut encoder = Encoder::new(&mut buf);
    let req = CreateQueue {
        header: MagmaCommandHeader {
            opcode: MAGMA_OPCODE_CREATE_QUEUE,
            size: std::mem::size_of::<CreateQueue>() as u32,
        },
        device: 0,
        address_space,
        info: *info,
    };
    encoder.encode_create_queue(&req)?;
    let len = encoder.bytes_written();
    submit(&buf[..len])
}

#[allow(dead_code)]
pub fn encode_and_submit_map_buffer_gpu<F>(
    address_space: u32,
    buffer: u32,
    buffer_offset: u64,
    gpu_va: u64,
    size: u64,
    flags: MagmaGpuMapFlags,
    submit: F,
) -> MagmaGpuResult<()>
where
    F: FnOnce(&[u8]) -> MagmaGpuResult<()>,
{
    let mut buf = [0u8; 256];
    let mut encoder = Encoder::new(&mut buf);
    let req = MapBufferGpu {
        header: MagmaCommandHeader {
            opcode: MAGMA_OPCODE_MAP_BUFFER_GPU,
            size: std::mem::size_of::<MapBufferGpu>() as u32,
        },
        address_space,
        buffer,
        buffer_offset,
        gpu_va,
        size,
        flags,
    };
    encoder.encode_map_buffer_gpu(&req)?;
    let len = encoder.bytes_written();
    submit(&buf[..len])
}

#[allow(dead_code)]
pub fn encode_and_submit_unmap_buffer_gpu<F>(
    address_space: u32,
    gpu_va: u64,
    size: u64,
    submit: F,
) -> MagmaGpuResult<()>
where
    F: FnOnce(&[u8]) -> MagmaGpuResult<()>,
{
    let mut buf = [0u8; 256];
    let mut encoder = Encoder::new(&mut buf);
    let req = UnmapBufferGpu {
        header: MagmaCommandHeader {
            opcode: MAGMA_OPCODE_UNMAP_BUFFER_GPU,
            size: std::mem::size_of::<UnmapBufferGpu>() as u32,
        },
        address_space,
        _pad0: 0,
        gpu_va,
        size,
    };
    encoder.encode_unmap_buffer_gpu(&req)?;
    let len = encoder.bytes_written();
    submit(&buf[..len])
}

#[allow(dead_code)]
pub fn encode_and_submit_submit_command<F>(
    queue: u32,
    submit_info: &MagmaSubmitInfo,
    submit: F,
) -> MagmaGpuResult<()>
where
    F: FnOnce(&[u8]) -> MagmaGpuResult<()>,
{
    let mut buf = [0u8; 512];
    let mut encoder = Encoder::new(&mut buf);
    let req = SubmitCommand {
        header: MagmaCommandHeader {
            opcode: MAGMA_OPCODE_SUBMIT_COMMAND,
            size: std::mem::size_of::<SubmitCommand>() as u32,
        },
        queue,
        ..Default::default()
    };
    encoder.encode_submit_command(&req)?;
    encoder.encode_submit_info(submit_info)?;
    if let Some(as_info) = submit_info.address_space_info() {
        encoder.encode_submit_address_space_info(as_info)?;
    } else if let Some(buf_info) = submit_info.buffer_info() {
        encoder.encode_submit_buffer_info(buf_info)?;
    }
    let len = encoder.bytes_written();
    submit(&buf[..len])
}

pub fn encode_and_submit_create_buffer<F>(
    create_info: &MagmaCreateBufferInfo,
    submit: F,
) -> MagmaGpuResult<()>
where
    F: FnOnce(&[u8]) -> MagmaGpuResult<()>,
{
    let mut buf = [0u8; 512];
    let mut encoder = Encoder::new(&mut buf);
    let req = CreateBuffer {
        header: MagmaCommandHeader {
            opcode: MAGMA_OPCODE_CREATE_BUFFER,
            size: std::mem::size_of::<CreateBuffer>() as u32,
        },
        device: 0,
        ..Default::default()
    };
    encoder.encode_create_buffer(&req)?;
    encoder.encode_create_buffer_info(create_info)?;
    let len = encoder.bytes_written();
    submit(&buf[..len])
}

#[allow(dead_code)]
pub fn encode_and_submit_create_sync_obj<F>(
    info: &MagmaCreateSyncObjInfo,
    submit: F,
) -> MagmaGpuResult<()>
where
    F: FnOnce(&[u8]) -> MagmaGpuResult<()>,
{
    let mut buf = [0u8; 256];
    let mut encoder = Encoder::new(&mut buf);
    let req = CreateSyncObj {
        header: MagmaCommandHeader {
            opcode: MAGMA_OPCODE_CREATE_SYNC_OBJ,
            size: std::mem::size_of::<CreateSyncObj>() as u32,
        },
        device: 0,
        ..Default::default()
    };
    encoder.encode_create_sync_obj(&req)?;
    encoder.encode_create_sync_obj_info(info)?;
    let len = encoder.bytes_written();
    submit(&buf[..len])
}

pub fn encode_and_submit_sync_obj_wait<F>(
    sync_obj: u32,
    timeout_ns: u64,
    submit: F,
) -> MagmaGpuResult<()>
where
    F: FnOnce(&[u8]) -> MagmaGpuResult<()>,
{
    let mut buf = [0u8; 256];
    let mut encoder = Encoder::new(&mut buf);
    let req = SyncObjWait {
        header: MagmaCommandHeader {
            opcode: MAGMA_OPCODE_SYNC_OBJ_WAIT,
            size: std::mem::size_of::<SyncObjWait>() as u32,
        },
        sync_obj,
        _pad0: 0,
        timeout_ns,
    };
    encoder.encode_sync_obj_wait(&req)?;
    let len = encoder.bytes_written();
    submit(&buf[..len])
}

pub fn encode_and_submit_sync_obj_signal<F>(sync_obj: u32, submit: F) -> MagmaGpuResult<()>
where
    F: FnOnce(&[u8]) -> MagmaGpuResult<()>,
{
    let mut buf = [0u8; 256];
    let mut encoder = Encoder::new(&mut buf);
    let req = SyncObjSignal {
        header: MagmaCommandHeader {
            opcode: MAGMA_OPCODE_SYNC_OBJ_SIGNAL,
            size: std::mem::size_of::<SyncObjSignal>() as u32,
        },
        sync_obj,
        _padding: 0,
    };
    encoder.encode_sync_obj_signal(&req)?;
    let len = encoder.bytes_written();
    submit(&buf[..len])
}

pub fn encode_and_submit_virt_create_render_thread<F>(
    thread_id: u32,
    ring_blob_id: u32,
    submit: F,
) -> MagmaGpuResult<()>
where
    F: FnOnce(&[u8]) -> MagmaGpuResult<()>,
{
    let mut buf = [0u8; 256];
    let mut encoder = Encoder::new(&mut buf);
    let req = VirtCreateRenderThread {
        header: MagmaCommandHeader {
            opcode: MAGMA_OPCODE_VIRT_CREATE_RENDER_THREAD,
            size: std::mem::size_of::<VirtCreateRenderThread>() as u32,
        },
        thread_id,
        ring_blob_id,
    };
    encoder.encode_virt_create_render_thread(&req)?;
    let len = encoder.bytes_written();
    submit(&buf[..len])
}

pub fn encode_and_submit_virt_ping<F>(thread_id: u32, submit: F) -> MagmaGpuResult<()>
where
    F: FnOnce(&[u8]) -> MagmaGpuResult<()>,
{
    let mut buf = [0u8; 256];
    let mut encoder = Encoder::new(&mut buf);
    let req = VirtPing {
        header: MagmaCommandHeader {
            opcode: MAGMA_OPCODE_VIRT_PING,
            size: std::mem::size_of::<VirtPing>() as u32,
        },
        thread_id,
        _padding: 0,
    };
    encoder.encode_virt_ping(&req)?;
    let len = encoder.bytes_written();
    submit(&buf[..len])
}

pub fn encode_and_submit_virt_create_fence<F>(
    ring_idx: u32,
    guest_sync_id: u32,
    submit: F,
) -> MagmaGpuResult<()>
where
    F: FnOnce(&[u8]) -> MagmaGpuResult<()>,
{
    let mut buf = [0u8; 256];
    let mut encoder = Encoder::new(&mut buf);
    let req = VirtCreateFence {
        header: MagmaCommandHeader {
            opcode: MAGMA_OPCODE_VIRT_CREATE_FENCE,
            size: std::mem::size_of::<VirtCreateFence>() as u32,
        },
        ring_idx,
        guest_sync_id,
    };
    encoder.encode_virt_create_fence(&req)?;
    let len = encoder.bytes_written();
    submit(&buf[..len])
}
