// Copyright 2026 The Magma GPU Project
// SPDX-License-Identifier: MIT

use magma_gpu::util::Result as MagmaGpuResult;
use std::sync::Arc;

use crate::magma::MagmaPhysicalDevice;
use crate::magma_defines::MagmaCreateBufferInfo;
use crate::magma_defines::MagmaMemoryType;
use crate::sys::windows::d3dkmt_common;
use crate::traits::PhysicalDevice;

pub trait VendorPrivateData: Send + Sync {
    fn createallocation_pdata(&self) -> Vec<u32> {
        Vec::new()
    }

    fn allocationinfo2_pdata(
        &self,
        _create_info: &MagmaCreateBufferInfo,
        _mem_types: &[MagmaMemoryType],
    ) -> Vec<u32> {
        Vec::new()
    }
}

pub fn enumerate_devices() -> MagmaGpuResult<Vec<MagmaPhysicalDevice>> {
    let mut devices: Vec<MagmaPhysicalDevice> = Vec::new();
    let adapters = d3dkmt_common::enumerate_adapters()?;

    for (adapter, info) in adapters {
        let physical_device: Arc<dyn PhysicalDevice> = Arc::new(adapter);
        devices.push(MagmaPhysicalDevice::new(physical_device, info));
    }

    Ok(devices)
}
