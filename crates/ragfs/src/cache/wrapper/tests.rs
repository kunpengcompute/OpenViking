// Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
// SPDX-License-Identifier: AGPL-3.0

use super::*;
use crate::cache_runtime::MemoryMockProvider;
use crate::plugins::MemFileSystem;

fn options() -> GrepOptions<'static> {
    GrepOptions {
        recursive: true,
        before_context: 1,
        after_context: 1,
        ..Default::default()
    }
}

async fn fixture(count: usize) -> (CachedFileSystem, Arc<MemoryMockProvider>, Vec<String>) {
    let backend = MemFileSystem::new();
    backend.mkdir("/docs", 0o755).await.unwrap();
    let mut paths = Vec::new();
    for index in 0..count {
        let path = format!("/docs/{index:04}.txt");
        backend
            .write(&path, b"before\nNeedle\nafter\n", 0, WriteFlag::Create)
            .await
            .unwrap();
        paths.push(path);
    }
    let provider = Arc::new(MemoryMockProvider::new());
    let fs = CachedFileSystem::with_runtime(
        Box::new(backend),
        CacheRuntime::memory_with_provider(provider.clone()),
        CacheNamespace::new("grep-batches"),
        CachePolicy::default().with_traversal_mode(CacheTraversalMode::CachedTraversal),
    );
    // Warm only bodies and directory entries, leaving the query bucket cold.
    for path in &paths {
        fs.read(path, 0, 0).await.unwrap();
    }
    fs.read_dir("/docs", None, None, None, None).await.unwrap();
    provider.reset_observed_reads();
    (fs, provider, paths)
}

async fn assert_matches_backend(fs: &CachedFileSystem, pattern: &str, options: GrepOptions<'_>) {
    let expected = fs.backend.grep("/docs", pattern, options).await.unwrap();
    let actual = fs.grep("/docs", pattern, options).await.unwrap();
    assert_eq!(
        serde_json::to_value(actual).unwrap(),
        serde_json::to_value(expected).unwrap()
    );
}

#[tokio::test]
async fn grep_fetches_payloads_in_256_key_batches_and_preserves_query_hits() {
    let (fs, provider, _) = fixture(513).await;
    assert_matches_backend(&fs, "Needle", options()).await;
    let payload_batches = provider
        .observed_read_batches()
        .into_iter()
        .filter(|keys| keys.iter().any(|key| key.contains(":file:")))
        .collect::<Vec<_>>();
    assert_eq!(
        payload_batches.iter().map(Vec::len).collect::<Vec<_>>(),
        [256, 256]
    );
    let reads = provider.observed_read_keys();
    assert_eq!(
        reads.iter().filter(|key| key.contains(":file:")).count(),
        513
    );
    let generations = reads
        .iter()
        .filter(|key| key.contains(":subtree:"))
        .collect::<Vec<_>>();
    assert_eq!(
        generations.len(),
        generations.iter().collect::<HashSet<_>>().len()
    );

    provider.reset_observed_reads();
    assert_matches_backend(&fs, "Needle", options()).await;
    assert!(!provider
        .observed_read_keys()
        .iter()
        .any(|key| key.contains(":file:")));
}

#[tokio::test]
async fn grep_small_limits_keep_256_key_fetches_and_stop_after_the_matching_batch() {
    let (fs, provider, _) = fixture(260).await;
    for limit in [1, 10] {
        provider.reset_observed_reads();
        let hits_before = fs.metrics().snapshot().file_hits;
        assert_matches_backend(
            &fs,
            "Needle",
            GrepOptions {
                node_limit: Some(limit),
                ..options()
            },
        )
        .await;
        let batches = provider
            .observed_read_batches()
            .into_iter()
            .filter(|keys| keys.iter().any(|key| key.contains(":file:")))
            .map(|keys| keys.len())
            .collect::<Vec<_>>();
        assert_eq!(batches, [256]);
        assert_eq!(fs.metrics().snapshot().file_hits - hits_before, 256);
    }
}

#[tokio::test]
async fn grep_small_limit_without_matches_scans_all_files_in_256_key_batches() {
    let (fs, provider, _) = fixture(513).await;
    assert_matches_backend(
        &fs,
        "absent",
        GrepOptions {
            node_limit: Some(10),
            ..options()
        },
    )
    .await;
    let batches = provider
        .observed_read_batches()
        .into_iter()
        .filter(|keys| keys.iter().any(|key| key.contains(":file:")))
        .map(|keys| keys.len())
        .collect::<Vec<_>>();
    assert_eq!(batches, [256, 256]);
    assert_eq!(fs.metrics().snapshot().file_hits, 513);
}

#[tokio::test]
async fn grep_batch_falls_back_for_missing_corrupt_wrong_path_and_stale_payloads() {
    let (fs, _, paths) = fixture(5).await;
    let before = fs.metrics().snapshot().backend_fallbacks;
    fs.runtime.del(&[fs.file_key(&paths[0])]).await.unwrap();
    fs.runtime
        .set(
            &fs.file_key(&paths[1]),
            Bytes::from_static(b"invalid"),
            SetOptions::default(),
        )
        .await
        .unwrap();
    let wrong = CacheEnvelope::file("/wrong".to_string(), b"wrong".to_vec(), Vec::new())
        .encode()
        .unwrap();
    fs.runtime
        .set(&fs.file_key(&paths[2]), wrong, SetOptions::default())
        .await
        .unwrap();
    // Leave the old body intact, as after an ancestor generation invalidation.
    fs.backend
        .write(
            &paths[3],
            b"changed\nNeedle\nnew context\n",
            0,
            WriteFlag::Truncate,
        )
        .await
        .unwrap();
    fs.bump_generation(&paths[3]).await;
    assert_matches_backend(&fs, "Needle", options()).await;
    assert_eq!(fs.metrics().snapshot().backend_fallbacks - before, 4);
}

#[tokio::test]
async fn grep_batch_preserves_policy_bypass_and_provider_failure_fallback() {
    let (mut fs, provider, paths) = fixture(4).await;
    fs.policy = fs.policy.with_bypass_prefix(&paths[0]);
    fs.backend
        .write(
            &paths[0],
            b"updated\nNeedle\nfresh\n",
            0,
            WriteFlag::Truncate,
        )
        .await
        .unwrap();
    assert_matches_backend(&fs, "Needle", options()).await;
    assert!(!provider
        .observed_read_keys()
        .contains(&fs.file_key(&paths[0])));
    provider.set_unavailable(true);
    assert_matches_backend(&fs, "(?i)needle", options()).await;
}

#[tokio::test]
async fn failed_payload_mget_bypasses_each_file_and_reads_backend_after_recovery() {
    let (fs, provider, paths) = fixture(3).await;
    provider.set_unavailable(true);
    let cached = fs
        .probe_grep_file_batch(&paths, &Mutex::new(HashMap::new()))
        .await;
    assert!(cached.iter().all(Option::is_none));
    provider.set_unavailable(false);
    for path in paths {
        assert!(fs.is_runtime_bypassed(&path).await);
        assert_eq!(
            fs.read_with_generation_cache(&path, 0, 0, &Mutex::new(HashMap::new()))
                .await
                .unwrap(),
            fs.backend.read(&path, 0, 0).await.unwrap()
        );
    }
}

#[tokio::test]
async fn grep_batch_keeps_limits_filters_and_context_semantics() {
    let (fs, _, paths) = fixture(260).await;
    for limit in [0, 1, 257, 300] {
        assert_matches_backend(
            &fs,
            "needle",
            GrepOptions {
                node_limit: Some(limit),
                case_insensitive: true,
                exclude_path: Some(&paths[0]),
                ..options()
            },
        )
        .await;
    }
    fs.mkdir("/docs/sub", 0o755).await.unwrap();
    fs.write(
        "/docs/sub/nested.txt",
        b"before\nNeedle\nafter\n",
        0,
        WriteFlag::Create,
    )
    .await
    .unwrap();
    for level_limit in [Some(1), Some(2), None] {
        for recursive in [false, true] {
            assert_matches_backend(
                &fs,
                "Needle",
                GrepOptions {
                    recursive,
                    level_limit,
                    ..options()
                },
            )
            .await;
        }
    }
}

#[tokio::test]
async fn grep_batch_observes_writes_renames_and_removals() {
    let (fs, _, paths) = fixture(3).await;
    assert_matches_backend(&fs, "Needle", options()).await;
    fs.write(&paths[0], b"no match", 0, WriteFlag::Truncate)
        .await
        .unwrap();
    assert_matches_backend(&fs, "Needle", options()).await;
    fs.rename(&paths[1], "/docs/renamed.txt").await.unwrap();
    assert_matches_backend(&fs, "Needle", options()).await;
    fs.remove(&paths[2]).await.unwrap();
    assert_matches_backend(&fs, "Needle", options()).await;
}

#[tokio::test]
async fn grep_payloads_share_the_provider_buffer_and_accept_empty_files() {
    let (fs, _, paths) = fixture(2).await;
    fs.write(&paths[1], b"", 0, WriteFlag::Truncate)
        .await
        .unwrap();
    let encoded = fs
        .runtime
        .get(&fs.file_key(&paths[0]))
        .await
        .unwrap()
        .unwrap();
    let cached = fs
        .probe_grep_file_batch(&paths, &Mutex::new(HashMap::new()))
        .await;
    let payload = cached[0].as_ref().unwrap();
    assert_eq!(payload.as_ref(), b"before\nNeedle\nafter\n");
    assert_eq!(
        payload.as_ptr(),
        encoded[encoded.len() - payload.len()..].as_ptr()
    );
    assert!(cached[1].as_ref().unwrap().is_empty());
    assert_matches_backend(&fs, "Needle", options()).await;
}

#[tokio::test]
async fn binary_cache_keys_leave_legacy_bodies_untouched_and_refill_from_backend() {
    let (fs, _, paths) = fixture(1).await;
    let key = fs.file_key(&paths[0]);
    assert!(key.starts_with("ragfs:v2:"));
    let old_key = key.replacen("ragfs:v2:", "ragfs:v1:", 1);
    let old = Bytes::from_static(br#"{"version":1,"kind":"File","path":"/docs/0000.txt","generations":[],"payload":{"File":[1]}}"#);
    fs.runtime
        .set(&old_key, old.clone(), SetOptions::default())
        .await
        .unwrap();
    fs.runtime.del(&[key.clone()]).await.unwrap();
    let before = fs.metrics().snapshot().backend_fallbacks;
    assert_eq!(
        fs.read(&paths[0], 0, 0).await.unwrap(),
        b"before\nNeedle\nafter\n"
    );
    assert_eq!(fs.metrics().snapshot().backend_fallbacks - before, 1);
    assert_eq!(fs.runtime.get(&old_key).await.unwrap(), Some(old.clone()));
    assert!(fs
        .runtime
        .get(&key)
        .await
        .unwrap()
        .unwrap()
        .starts_with(b"RGFC\x02"));

    // A legacy or corrupt value accidentally stored under a v2 key is a miss too.
    fs.runtime
        .set(&key, old, SetOptions::default())
        .await
        .unwrap();
    let before = fs.metrics().snapshot().backend_fallbacks;
    assert_eq!(
        fs.read(&paths[0], 0, 0).await.unwrap(),
        b"before\nNeedle\nafter\n"
    );
    assert_eq!(fs.metrics().snapshot().backend_fallbacks - before, 1);
    assert!(fs
        .runtime
        .get(&key)
        .await
        .unwrap()
        .unwrap()
        .starts_with(b"RGFC\x02"));
}

#[tokio::test]
async fn cached_tree_reuses_generations_and_preserves_sorting_pagination_and_invalidation() {
    let (fs, provider, _) = fixture(2).await;
    for path in ["/docs/a", "/docs/a/deep", "/docs/b"] {
        fs.mkdir(path, 0o755).await.unwrap();
    }
    for path in [
        "/docs/a/deep/nested.txt",
        "/docs/b/last.txt",
        "/docs/.hidden",
    ] {
        fs.write(path, b"Needle", 0, WriteFlag::Create)
            .await
            .unwrap();
    }
    for path in ["/docs", "/docs/a", "/docs/a/deep", "/docs/b"] {
        fs.read_dir(path, None, None, None, None).await.unwrap();
    }
    for (node_limit, level_limit, offset, sort_by, sort_order, directories_only) in [
        (None, None, None, None, None, false),
        (
            Some(3),
            None,
            Some(1),
            Some(ListSortBy::Name),
            Some(SortOrder::Desc),
            false,
        ),
        (
            None,
            Some(1),
            None,
            Some(ListSortBy::Mtime),
            Some(SortOrder::Asc),
            false,
        ),
        (
            Some(2),
            None,
            None,
            Some(ListSortBy::Name),
            Some(SortOrder::Asc),
            true,
        ),
    ] {
        provider.reset_observed_reads();
        let expected = fs
            .backend
            .tree_directory(
                "/docs",
                false,
                node_limit,
                level_limit,
                offset,
                sort_by,
                sort_order,
                directories_only,
            )
            .await
            .unwrap();
        let actual = fs
            .tree_directory(
                "/docs",
                false,
                node_limit,
                level_limit,
                offset,
                sort_by,
                sort_order,
                directories_only,
            )
            .await
            .unwrap();
        assert_eq!(
            serde_json::to_value(actual).unwrap(),
            serde_json::to_value(expected).unwrap()
        );
        let reads = provider.observed_read_keys();
        let generations = reads
            .iter()
            .filter(|key| key.contains(":subtree:"))
            .collect::<Vec<_>>();
        assert!(!generations.is_empty());
        assert_eq!(
            generations.len(),
            generations.iter().collect::<HashSet<_>>().len()
        );
    }
    fs.remove_all("/docs/a").await.unwrap();
    let expected = fs
        .backend
        .tree_directory("/docs", false, None, None, None, None, None, false)
        .await
        .unwrap();
    let actual = fs
        .tree_directory("/docs", false, None, None, None, None, None, false)
        .await
        .unwrap();
    assert_eq!(
        serde_json::to_value(actual).unwrap(),
        serde_json::to_value(expected).unwrap()
    );
}
