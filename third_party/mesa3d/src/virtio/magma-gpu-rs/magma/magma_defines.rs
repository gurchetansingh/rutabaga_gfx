// Copyright 2026 The Magma GPU Project
// SPDX-License-Identifier: MIT

use magma_gpu::util::Error as MagmaGpuError;
use remain::sorted;
use thiserror::Error;
use zerocopy::FromBytes;
use zerocopy::IntoBytes;

pub use crate::protocol::*;

/// An error type based on magma_common_defs.h
#[sorted]
#[derive(Error, Debug)]
pub enum MagmaError {
    #[error("Access Denied")]
    AccessDenied,
    #[error("Bad State")]
    BadState,
    #[error("Connection Lost")]
    ConnectionLost,
    #[error("Context Killed")]
    ContextKilled,
    #[error("Internal Error")]
    InternalError,
    #[error("Invalid Arguments")]
    InvalidArgs,
    #[error("A Mesa error was returned {0}")]
    MagmaError(MagmaGpuError),
    #[error("Memory Error")]
    MemoryError,
    #[error("Timed out")]
    TimedOut,
    #[error("Unimplemented")]
    Unimplemented,
}

impl From<MagmaGpuError> for MagmaError {
    fn from(e: MagmaGpuError) -> MagmaError {
        MagmaError::MagmaError(e)
    }
}

impl From<MagmaError> for MagmaStatus {
    fn from(e: MagmaError) -> Self {
        match e {
            MagmaError::AccessDenied => MagmaStatus::AccessDenied,
            MagmaError::BadState => MagmaStatus::InternalError,
            MagmaError::ConnectionLost => MagmaStatus::InternalError,
            MagmaError::ContextKilled => MagmaStatus::ContextKilled,
            MagmaError::InternalError => MagmaStatus::InternalError,
            MagmaError::InvalidArgs => MagmaStatus::InvalidArgs,
            MagmaError::MemoryError => MagmaStatus::MemoryError,
            MagmaError::TimedOut => MagmaStatus::TimedOut,
            MagmaError::Unimplemented => MagmaStatus::Unimplemented,
            MagmaError::MagmaError(me) => match me {
                MagmaGpuError::Unsupported => MagmaStatus::Unimplemented,
                _ => MagmaStatus::InternalError,
            },
        }
    }
}

impl From<MagmaError> for i32 {
    fn from(e: MagmaError) -> i32 {
        MagmaStatus::from(e) as i32
    }
}

pub type MagmaResult<T> = std::result::Result<T, MagmaError>;

pub const MAGMA_BUS_TYPE_UNKNOWN: u32 = 0;
pub const MAGMA_BUS_TYPE_PCI: u32 = 1;
pub const MAGMA_BUS_TYPE_PLATFORM: u32 = 2;

// Handled via pub use crate::protocol::*;

// Should be set in the case of VRAM only
pub const MAGMA_HEAP_DEVICE_LOCAL_BIT: u64 = 0x00000001;
pub const MAGMA_HEAP_CPU_VISIBLE_BIT: u64 = 0x00000010;
impl MagmaHeap {
    pub fn is_device_local(&self) -> bool {
        self.heap_flags & MAGMA_HEAP_DEVICE_LOCAL_BIT != 0
    }

    pub fn is_cpu_visible(&self) -> bool {
        self.heap_flags & MAGMA_HEAP_CPU_VISIBLE_BIT != 0
    }
}

pub const MAGMA_MEMORY_PROPERTY_DEVICE_LOCAL_BIT: u32 = 0x00000001;
pub const MAGMA_MEMORY_PROPERTY_HOST_VISIBLE_BIT: u32 = 0x00000002;
pub const MAGMA_MEMORY_PROPERTY_HOST_COHERENT_BIT: u32 = 0x00000004;
pub const MAGMA_MEMORY_PROPERTY_HOST_CACHED_BIT: u32 = 0x00000008;
pub const MAGMA_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT: u32 = 0x00000010;
pub const MAGMA_MEMORY_PROPERTY_PROTECTED_BIT: u32 = 0x00000020;

impl MagmaMemoryType {
    pub fn is_device_local(&self) -> bool {
        self.property_flags & MAGMA_MEMORY_PROPERTY_DEVICE_LOCAL_BIT != 0
    }

    pub fn is_coherent(&self) -> bool {
        self.property_flags & MAGMA_MEMORY_PROPERTY_HOST_COHERENT_BIT != 0
    }

    pub fn is_cached(&self) -> bool {
        self.property_flags & MAGMA_MEMORY_PROPERTY_HOST_CACHED_BIT != 0
    }

    pub fn is_protected(&self) -> bool {
        self.property_flags & MAGMA_MEMORY_PROPERTY_PROTECTED_BIT != 0
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MagmaMemoryProperties {
    pub memory_type_count: u32,
    pub memory_heap_count: u32,
    pub memory_types: *mut MagmaMemoryType,
    pub memory_heaps: *mut MagmaHeap,
}

impl Default for MagmaMemoryProperties {
    fn default() -> Self {
        Self {
            memory_type_count: 0,
            memory_heap_count: 0,
            memory_types: core::ptr::null_mut(),
            memory_heaps: core::ptr::null_mut(),
        }
    }
}

// Common allocation flags
//  - MAGMA_BUFFER_FLAG_EXTERNAL: The buffer *may* be exported as an OS-specific handle
//  - MAGMA_BUFFER_FLAG_SCANOUT: The buffer *may* be used by the scanout engine directly
pub const MAGMA_BUFFER_FLAG_EXTERNAL: u32 = 0x000000001;
pub const MAGMA_BUFFER_FLAG_SCANOUT: u32 = 0x000000002;

// Acceptable buffer vendor flags if the vendor is AMD:
//  - MAGMA_BUFFER_FLAG_AMD_FLAG_OA: Ordered append, used by 3D/Compute engines
//  - MAGMA_BUFFER_FLAG_AMD_FLAG_GDS: Global on-chip data storage. Used to share
//                                    data across shader threads
pub const MAGMA_BUFFER_FLAG_AMD_OA: u32 = 0x000000001;
pub const MAGMA_BUFFER_FLAG_AMD_GDS: u32 = 0x000000002;

pub const MAGMA_SYNC_WHOLE_RANGE: u64 = 1 << 0;
pub const MAGMA_SYNC_RANGES: u64 = 1 << 1;
pub const MAGMA_SYNC_INVALIDATE_READ: u64 = 1 << 2;
pub const MAGMA_SYNC_INVALIDATE_WRITE: u64 = 1 << 3;

pub type MagmaSyncType = MagmaSyncObjType;
pub const MAGMA_SYNC_OBJ_TYPE_BINARY: u32 = MagmaSyncObjType::Binary as u32;
pub const MAGMA_SYNC_OBJ_TYPE_TIMELINE: u32 = MagmaSyncObjType::Timeline as u32;

impl From<u32> for MagmaSyncType {
    fn from(val: u32) -> Self {
        match val {
            1 => MagmaSyncType::Timeline,
            _ => MagmaSyncType::Binary,
        }
    }
}

#[repr(C)]
#[derive(Clone, Default, Debug, IntoBytes, FromBytes)]
pub struct MagmaMappedMemoryRange {
    pub offset: u64,
    pub size: u64,
}

// Same as PCI id
pub const MAGMA_VENDOR_ID_INTEL: u16 = 0x8086;
pub const MAGMA_VENDOR_ID_AMD: u16 = 0x1002;
pub const MAGMA_VENDOR_ID_MALI: u16 = 0x13B5;
pub const MAGMA_VENDOR_ID_QCOM: u16 = 0x5413;
pub const MAGMA_VENDOR_ID_VIRTGPU: u16 = 0x1AF4;

use magma_gpu::util::Handle as MagmaGpuHandle;

pub struct MagmaImportHandleInfo {
    pub handle: MagmaGpuHandle,
    pub size: u64,
    pub memory_type_idx: u32,
}
