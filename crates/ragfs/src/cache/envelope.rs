// Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
// SPDX-License-Identifier: AGPL-3.0

//! Stable values stored behind the provider contract.

use crate::cache_runtime::{CacheError, CacheResult};
use crate::core::FileInfo;
use bytes::Bytes;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

const CACHE_ENVELOPE_MAGIC: &[u8; 4] = b"RGFC";
const CACHE_ENVELOPE_VERSION: u8 = 2;
const KIND_FILE: u8 = 1;
const KIND_DIRECTORY: u8 = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CacheObjectKind {
    File,
    Directory,
}

#[derive(Debug, Clone)]
pub(crate) struct GenerationSnapshot {
    pub key: String,
    pub value: u64,
}

#[derive(Debug, Clone)]
enum CachePayload {
    File(Bytes),
    Directory(Vec<FileInfo>),
}

#[derive(Debug, Clone)]
pub(crate) struct CacheEnvelope {
    kind: CacheObjectKind,
    path: String,
    generations: Vec<GenerationSnapshot>,
    payload: CachePayload,
}

impl CacheEnvelope {
    pub fn file(path: String, data: Vec<u8>, generations: Vec<GenerationSnapshot>) -> Self {
        Self {
            kind: CacheObjectKind::File,
            path,
            generations,
            payload: CachePayload::File(Bytes::from(data)),
        }
    }

    pub fn directory(
        path: String,
        entries: Vec<FileInfo>,
        generations: Vec<GenerationSnapshot>,
    ) -> Self {
        Self {
            kind: CacheObjectKind::Directory,
            path,
            generations,
            payload: CachePayload::Directory(entries),
        }
    }

    pub fn encode(&self) -> CacheResult<Bytes> {
        let mut encoded = Vec::new();
        encoded.extend_from_slice(CACHE_ENVELOPE_MAGIC);
        encoded.push(CACHE_ENVELOPE_VERSION);
        encoded.push(match self.kind {
            CacheObjectKind::File => KIND_FILE,
            CacheObjectKind::Directory => KIND_DIRECTORY,
        });
        write_string(&mut encoded, &self.path)?;
        write_u32(&mut encoded, self.generations.len())?;
        for snapshot in &self.generations {
            write_string(&mut encoded, &snapshot.key)?;
            encoded.extend_from_slice(&snapshot.value.to_be_bytes());
        }
        match &self.payload {
            CachePayload::File(data) => {
                write_u64(&mut encoded, data.len())?;
                encoded.extend_from_slice(data);
            }
            CachePayload::Directory(entries) => {
                let payload = encode_directory_payload(entries)?;
                write_u64(&mut encoded, payload.len())?;
                encoded.extend_from_slice(&payload);
            }
        }
        Ok(Bytes::from(encoded))
    }

    pub fn decode(value: Bytes) -> CacheResult<Self> {
        let mut reader = BinaryReader::new(&value);
        let magic = reader.read_bytes(CACHE_ENVELOPE_MAGIC.len())?;
        if magic != CACHE_ENVELOPE_MAGIC {
            return Err(CacheError::InvalidData(
                "invalid envelope magic".to_string(),
            ));
        }
        let version = reader.read_u8()?;
        if version != CACHE_ENVELOPE_VERSION {
            return Err(CacheError::InvalidData(format!(
                "unsupported envelope version {version}"
            )));
        }
        let kind = match reader.read_u8()? {
            KIND_FILE => CacheObjectKind::File,
            KIND_DIRECTORY => CacheObjectKind::Directory,
            other => {
                return Err(CacheError::InvalidData(format!(
                    "unsupported envelope kind {other}"
                )))
            }
        };
        let path = reader.read_string()?;
        let generation_count = reader.read_count(12)?;
        let mut generations = Vec::with_capacity(generation_count);
        for _ in 0..generation_count {
            generations.push(GenerationSnapshot {
                key: reader.read_string()?,
                value: reader.read_u64()?,
            });
        }
        let payload_len = usize::try_from(reader.read_u64()?)
            .map_err(|_| CacheError::InvalidData("cache envelope payload too large".to_string()))?;
        let payload_start = reader.position;
        let payload_bytes = reader.read_bytes(payload_len)?;
        if !reader.is_finished() {
            return Err(CacheError::InvalidData(
                "trailing bytes in cache envelope".to_string(),
            ));
        }
        let payload = match kind {
            CacheObjectKind::File => {
                CachePayload::File(value.slice(payload_start..payload_start + payload_len))
            }
            CacheObjectKind::Directory => {
                CachePayload::Directory(decode_directory_payload(payload_bytes)?)
            }
        };
        Ok(Self {
            kind,
            path,
            generations,
            payload,
        })
    }

    pub fn matches(&self, kind: CacheObjectKind, path: &str) -> bool {
        self.kind == kind && self.path == path
    }

    pub fn generations(&self) -> &[GenerationSnapshot] {
        &self.generations
    }

    pub fn into_file(self) -> CacheResult<Bytes> {
        match self.payload {
            CachePayload::File(data) => Ok(data),
            CachePayload::Directory(_) => {
                Err(CacheError::InvalidData("expected file payload".to_string()))
            }
        }
    }

    pub fn into_directory(self) -> CacheResult<Vec<FileInfo>> {
        match self.payload {
            CachePayload::Directory(entries) => Ok(entries),
            CachePayload::File(_) => Err(CacheError::InvalidData(
                "expected directory payload".to_string(),
            )),
        }
    }
}

fn write_u32(buffer: &mut Vec<u8>, value: usize) -> CacheResult<()> {
    let value = u32::try_from(value)
        .map_err(|_| CacheError::InvalidData("cache envelope field too large".to_string()))?;
    buffer.extend_from_slice(&value.to_be_bytes());
    Ok(())
}

fn write_u64(buffer: &mut Vec<u8>, value: usize) -> CacheResult<()> {
    let value = u64::try_from(value)
        .map_err(|_| CacheError::InvalidData("cache envelope payload too large".to_string()))?;
    buffer.extend_from_slice(&value.to_be_bytes());
    Ok(())
}

fn write_string(buffer: &mut Vec<u8>, value: &str) -> CacheResult<()> {
    write_u32(buffer, value.len())?;
    buffer.extend_from_slice(value.as_bytes());
    Ok(())
}

fn encode_directory_payload(entries: &[FileInfo]) -> CacheResult<Vec<u8>> {
    let mut encoded = Vec::new();
    write_u32(&mut encoded, entries.len())?;
    for entry in entries {
        write_string(&mut encoded, &entry.name)?;
        encoded.extend_from_slice(&entry.size.to_be_bytes());
        encoded.extend_from_slice(&entry.mode.to_be_bytes());
        let (secs, nanos) = system_time_to_parts(entry.mod_time)?;
        encoded.extend_from_slice(&secs.to_be_bytes());
        encoded.extend_from_slice(&nanos.to_be_bytes());
        encoded.push(u8::from(entry.is_dir));
    }
    Ok(encoded)
}

fn decode_directory_payload(value: &[u8]) -> CacheResult<Vec<FileInfo>> {
    let mut reader = BinaryReader::new(value);
    let entry_count = reader.read_count(29)?;
    let mut entries = Vec::with_capacity(entry_count);
    for _ in 0..entry_count {
        let name = reader.read_string()?;
        let size = reader.read_u64()?;
        let mode = reader.read_u32()?;
        let secs = reader.read_i64()?;
        let nanos = reader.read_u32()?;
        if nanos >= 1_000_000_000 {
            return Err(CacheError::InvalidData(
                "invalid FileInfo timestamp nanos".to_string(),
            ));
        }
        let is_dir = match reader.read_u8()? {
            0 => false,
            1 => true,
            _ => {
                return Err(CacheError::InvalidData(
                    "invalid FileInfo directory flag".to_string(),
                ))
            }
        };
        entries.push(FileInfo {
            name,
            size,
            mode,
            mod_time: parts_to_system_time(secs, nanos)?,
            is_dir,
        });
    }
    if !reader.is_finished() {
        return Err(CacheError::InvalidData(
            "trailing bytes in directory payload".to_string(),
        ));
    }
    Ok(entries)
}

// Use signed Unix seconds with a nonnegative nanosecond fraction, including
// timestamps before the epoch (e.g. -0.5s is encoded as -1s + 500,000,000ns).
fn system_time_to_parts(value: SystemTime) -> CacheResult<(i64, u32)> {
    let parts = match value.duration_since(UNIX_EPOCH) {
        Ok(duration) => i64::try_from(duration.as_secs())
            .ok()
            .map(|secs| (secs, duration.subsec_nanos())),
        Err(error) => {
            let duration = error.duration();
            let nanos = duration.subsec_nanos();
            i64::try_from(duration.as_secs())
                .ok()
                .and_then(|secs| secs.checked_neg())
                .and_then(|secs| {
                    if nanos == 0 {
                        Some((secs, 0))
                    } else {
                        secs.checked_sub(1)
                            .map(|secs| (secs, 1_000_000_000 - nanos))
                    }
                })
        }
    };
    parts.ok_or_else(|| CacheError::InvalidData("FileInfo timestamp out of range".to_string()))
}

fn parts_to_system_time(secs: i64, nanos: u32) -> CacheResult<SystemTime> {
    let seconds = Duration::from_secs(secs.unsigned_abs());
    let base = if secs >= 0 {
        UNIX_EPOCH.checked_add(seconds)
    } else {
        UNIX_EPOCH.checked_sub(seconds)
    };
    base.and_then(|time| time.checked_add(Duration::from_nanos(u64::from(nanos))))
        .ok_or_else(|| CacheError::InvalidData("FileInfo timestamp out of range".to_string()))
}

struct BinaryReader<'a> {
    value: &'a [u8],
    position: usize,
}

impl<'a> BinaryReader<'a> {
    fn new(value: &'a [u8]) -> Self {
        Self { value, position: 0 }
    }

    fn is_finished(&self) -> bool {
        self.position == self.value.len()
    }

    fn read_bytes(&mut self, len: usize) -> CacheResult<&'a [u8]> {
        let end = self
            .position
            .checked_add(len)
            .ok_or_else(|| CacheError::InvalidData("cache envelope offset overflow".to_string()))?;
        if end > self.value.len() {
            return Err(CacheError::InvalidData(
                "truncated cache envelope".to_string(),
            ));
        }
        let bytes = &self.value[self.position..end];
        self.position = end;
        Ok(bytes)
    }

    fn read_count(&mut self, min_item_size: usize) -> CacheResult<usize> {
        let count = self.read_u32()? as usize;
        if count > (self.value.len() - self.position) / min_item_size {
            return Err(CacheError::InvalidData(
                "invalid cache envelope item count".to_string(),
            ));
        }
        Ok(count)
    }

    fn read_u8(&mut self) -> CacheResult<u8> {
        Ok(self.read_bytes(1)?[0])
    }

    fn read_u32(&mut self) -> CacheResult<u32> {
        let bytes: [u8; 4] = self
            .read_bytes(4)?
            .try_into()
            .map_err(|_| CacheError::InvalidData("invalid u32 in cache envelope".to_string()))?;
        Ok(u32::from_be_bytes(bytes))
    }

    fn read_u64(&mut self) -> CacheResult<u64> {
        let bytes: [u8; 8] = self
            .read_bytes(8)?
            .try_into()
            .map_err(|_| CacheError::InvalidData("invalid u64 in cache envelope".to_string()))?;
        Ok(u64::from_be_bytes(bytes))
    }

    fn read_i64(&mut self) -> CacheResult<i64> {
        let bytes: [u8; 8] = self
            .read_bytes(8)?
            .try_into()
            .map_err(|_| CacheError::InvalidData("invalid i64 in cache envelope".to_string()))?;
        Ok(i64::from_be_bytes(bytes))
    }

    fn read_string(&mut self) -> CacheResult<String> {
        let len = self.read_u32()? as usize;
        let bytes = self.read_bytes(len)?;
        std::str::from_utf8(bytes)
            .map(str::to_owned)
            .map_err(|error| CacheError::InvalidData(error.to_string()))
    }
}

#[cfg(test)]
mod tests;
