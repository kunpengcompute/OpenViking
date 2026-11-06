// Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
// SPDX-License-Identifier: AGPL-3.0

use super::*;

fn generations() -> Vec<GenerationSnapshot> {
    vec![GenerationSnapshot {
        key: "ragfs:v2:test:subtree:0123456789abcdef".to_string(),
        value: u64::MAX,
    }]
}

#[test]
fn binary_files_round_trip_and_borrow_the_encoded_payload() {
    for data in [Vec::new(), (0..=255).collect::<Vec<u8>>(), vec![b'x'; 1024]] {
        let encoded = CacheEnvelope::file("/文档/a.txt".into(), data.clone(), generations())
            .encode()
            .unwrap();
        assert!(encoded.starts_with(b"RGFC\x02\x01"));
        let decoded = CacheEnvelope::decode(encoded.clone()).unwrap();
        assert!(decoded.matches(CacheObjectKind::File, "/文档/a.txt"));
        assert_eq!(decoded.generations()[0].key, generations()[0].key);
        assert_eq!(decoded.generations()[0].value, u64::MAX);
        let payload = decoded.into_file().unwrap();
        assert_eq!(payload.as_ref(), data);
        if !data.is_empty() {
            assert_eq!(
                payload.as_ptr(),
                encoded[encoded.len() - data.len()..].as_ptr()
            );
        }
        if data.len() == 1024 {
            let json = serde_json::to_vec(&serde_json::json!({
                "version": 1, "kind": "File", "path": "/文档/a.txt",
                "generations": [{"key": generations()[0].key, "value": u64::MAX}],
                "payload": {"File": data}
            }))
            .unwrap();
            assert!(encoded.len() * 3 < json.len());
        }
    }
}

#[test]
fn binary_directories_preserve_metadata_and_subsecond_timestamps() {
    let times = [
        UNIX_EPOCH,
        UNIX_EPOCH + Duration::new(123, 987_654_321),
        UNIX_EPOCH - Duration::from_millis(500),
        UNIX_EPOCH - Duration::new(123, 987_654_321),
    ];
    let entries = times
        .into_iter()
        .enumerate()
        .map(|(index, time)| {
            FileInfo::new(
                format!(".目录-{index}"),
                u64::MAX,
                u32::MAX,
                time,
                index % 2 == 0,
            )
        })
        .collect::<Vec<_>>();
    for entries in [Vec::new(), entries] {
        let encoded = CacheEnvelope::directory("/docs".into(), entries.clone(), generations())
            .encode()
            .unwrap();
        assert!(encoded.starts_with(b"RGFC\x02\x02"));
        let decoded = CacheEnvelope::decode(encoded).unwrap();
        assert!(decoded.matches(CacheObjectKind::Directory, "/docs"));
        assert_eq!(decoded.generations()[0].value, u64::MAX);
        let actual = decoded.into_directory().unwrap();
        assert_eq!(actual.len(), entries.len());
        for (actual, expected) in actual.iter().zip(entries) {
            assert_eq!(actual.name, expected.name);
            assert_eq!(actual.size, expected.size);
            assert_eq!(actual.mode, expected.mode);
            assert_eq!(actual.mod_time, expected.mod_time);
            assert_eq!(actual.is_dir, expected.is_dir);
        }
    }
}

#[test]
fn malformed_envelopes_are_rejected_without_panicking_or_large_allocations() {
    let file = CacheEnvelope::file("/a".into(), b"payload".to_vec(), generations())
        .encode()
        .unwrap();
    let directory = CacheEnvelope::directory(
        "/a".into(),
        vec![FileInfo::new("b".into(), 1, 0o644, UNIX_EPOCH, false)],
        Vec::new(),
    )
    .encode()
    .unwrap();
    for encoded in [&file, &directory] {
        for length in 0..encoded.len() {
            assert!(CacheEnvelope::decode(encoded.slice(..length)).is_err());
        }
        let mut trailing = encoded.to_vec();
        trailing.push(0);
        assert!(CacheEnvelope::decode(trailing.into()).is_err());
    }
    let mut invalid = Vec::new();
    for (index, value) in [(0, b'?'), (4, 1), (5, 0), (10, 0xff)] {
        let mut bytes = file.to_vec();
        bytes[index] = value;
        invalid.push(bytes);
    }
    let mut huge_generations = file.to_vec();
    huge_generations[12..16].copy_from_slice(&u32::MAX.to_be_bytes());
    invalid.push(huge_generations);
    let mut huge_payload = file.to_vec();
    let payload_length_offset = file.len() - b"payload".len() - 8;
    huge_payload[payload_length_offset..payload_length_offset + 8]
        .copy_from_slice(&u64::MAX.to_be_bytes());
    invalid.push(huge_payload);
    let mut huge_directory = directory.to_vec();
    // This fixture has a two-byte path, no generations and a four-byte entry count.
    huge_directory[24..28].copy_from_slice(&u32::MAX.to_be_bytes());
    invalid.push(huge_directory);
    let mut bad_nanos = directory.to_vec();
    let nanos_offset = bad_nanos.len() - 5;
    bad_nanos[nanos_offset..nanos_offset + 4].copy_from_slice(&1_000_000_000u32.to_be_bytes());
    invalid.push(bad_nanos);
    let mut bad_flag = directory.to_vec();
    *bad_flag.last_mut().unwrap() = 2;
    invalid.push(bad_flag);
    invalid.push(
        br#"{"version":1,"kind":"File","path":"/a","generations":[],"payload":{"File":[1]}}"#
            .to_vec(),
    );
    for value in invalid {
        assert!(CacheEnvelope::decode(value.into()).is_err());
    }
    assert!(CacheEnvelope::decode(file)
        .unwrap()
        .into_directory()
        .is_err());
    assert!(CacheEnvelope::decode(directory)
        .unwrap()
        .into_file()
        .is_err());
}
