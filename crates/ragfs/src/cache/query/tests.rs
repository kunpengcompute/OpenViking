// Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
// SPDX-License-Identifier: AGPL-3.0

use super::*;
use crate::cache::{CacheNamespace, CachePolicy, CacheTraversalMode, CachedFileSystem};
use crate::cache_runtime::MemoryMockProvider;
use crate::core::{FileSystem, GrepOptions, ListSortBy, SortOrder, WriteFlag};
use crate::plugins::memfs::MemFileSystem;
use serde_json::json;
use tokio::sync::oneshot;

fn fixture() -> (Arc<QueryCache>, Arc<CacheRuntime>, Arc<MemoryMockProvider>) {
    let provider = Arc::new(MemoryMockProvider::new());
    let runtime = CacheRuntime::memory_with_provider(provider.clone());
    let cache = Arc::new(QueryCache::new(
        "queries",
        1,
        runtime.clone(),
        Arc::new(CacheMetrics::default()),
    ));
    (cache, runtime, provider)
}

async fn query(cache: &QueryCache, path: &str, value: usize) -> usize {
    cache
        .get_or_load(path, "stat", json!([]), async { Ok(value) })
        .await
        .unwrap()
}

#[tokio::test]
async fn buckets_keep_operations_and_complete_parameters_separate() {
    let (cache, _, _) = fixture();
    for op in ["stat", "tree_directory", "glob_directory", "grep"] {
        for params in [json!([]), json!([null]), json!([0]), json!([1, "a"])] {
            let expected = format!("{op}:{params}");
            let first = cache
                .get_or_load("/docs", op, params.clone(), async { Ok(expected.clone()) })
                .await
                .unwrap();
            let hit: String = cache
                .get_or_load("docs/", op, params, async { panic!("cache miss") })
                .await
                .unwrap();
            assert_eq!(first, expected);
            assert_eq!(hit, expected);
        }
    }
}

#[tokio::test]
async fn invalidation_deletes_ancestors_and_descendants_but_preserves_bodies_and_siblings() {
    let (cache, runtime, provider) = fixture();
    let paths = [
        "/",
        "/docs",
        "/docs/a",
        "/docs/a/child",
        "/docs/ab",
        "/other",
    ];
    for path in paths {
        assert_eq!(query(&cache, path, 1).await, 1);
    }
    let body_key = "ragfs:v1:queries:file:body";
    runtime
        .set(body_key, Bytes::from_static(b"body"), SetOptions::default())
        .await
        .unwrap();
    cache.invalidate("/docs/a").await;
    assert_eq!(runtime.get(body_key).await.unwrap().unwrap(), b"body"[..]);
    let keys = provider.keys().await;
    for path in ["/", "/docs", "/docs/a", "/docs/a/child"] {
        assert!(!keys.contains(&cache.key(path)));
        assert_eq!(query(&cache, path, 2).await, 2);
    }
    for path in ["/docs/ab", "/other"] {
        assert_eq!(query(&cache, path, 2).await, 1);
    }
    cache.invalidate("/").await;
    for path in paths {
        assert_eq!(query(&cache, path, 3).await, 3);
    }
    assert_eq!(runtime.get(body_key).await.unwrap().unwrap(), b"body"[..]);
}

#[tokio::test]
async fn invalidation_prevents_inflight_query_from_refilling_old_results() {
    let (cache, _, _) = fixture();
    let (started_tx, started_rx) = oneshot::channel();
    let (resume_tx, resume_rx) = oneshot::channel();
    let worker = cache.clone();
    let task = tokio::spawn(async move {
        worker
            .get_or_load("/docs/a", "stat", json!([]), async {
                started_tx.send(()).unwrap();
                resume_rx.await.unwrap();
                Ok(1usize)
            })
            .await
            .unwrap()
    });
    started_rx.await.unwrap();
    cache.invalidate("/docs").await;
    resume_tx.send(()).unwrap();
    assert_eq!(task.await.unwrap(), 1);
    assert_eq!(query(&cache, "/docs/a", 2).await, 2);
    assert_eq!(query(&cache, "/docs/a", 3).await, 2);
}

#[tokio::test]
async fn concurrent_parameter_variants_do_not_overwrite_each_other() {
    let (cache, _, _) = fixture();
    let barrier = Arc::new(tokio::sync::Barrier::new(2));
    let mut tasks = Vec::new();
    for variant in [1usize, 2] {
        let worker = cache.clone();
        let barrier = barrier.clone();
        tasks.push(tokio::spawn(async move {
            worker
                .get_or_load("/docs", "glob_directory", json!([variant]), async {
                    barrier.wait().await;
                    Ok(variant)
                })
                .await
                .unwrap()
        }));
    }
    for task in tasks {
        task.await.unwrap();
    }
    for variant in [1usize, 2] {
        let hit: usize = cache
            .get_or_load("/docs", "glob_directory", json!([variant]), async {
                panic!("concurrent fill lost a query variant")
            })
            .await
            .unwrap();
        assert_eq!(hit, variant);
    }
}

#[tokio::test]
async fn failed_delete_bypasses_stale_queries_without_touching_body_cache() {
    let (cache, runtime, provider) = fixture();
    assert_eq!(query(&cache, "/docs", 1).await, 1);
    runtime
        .set("body", Bytes::from_static(b"kept"), SetOptions::default())
        .await
        .unwrap();
    provider.set_delete_failure(true);
    cache.invalidate("/docs").await;
    provider.set_delete_failure(false);
    assert_eq!(query(&cache, "/docs", 2).await, 2);
    assert_eq!(query(&cache, "/docs", 3).await, 3);
    assert_eq!(runtime.get("body").await.unwrap().unwrap(), b"kept"[..]);
}

#[tokio::test]
async fn corrupt_buckets_and_provider_errors_fall_back_to_backend() {
    let (cache, runtime, provider) = fixture();
    runtime
        .set(
            &cache.key("/docs"),
            Bytes::from_static(b"invalid"),
            SetOptions::default(),
        )
        .await
        .unwrap();
    assert_eq!(query(&cache, "/docs", 1).await, 1);
    assert_eq!(query(&cache, "/docs", 2).await, 1);
    provider.set_unavailable(true);
    assert_eq!(query(&cache, "/docs", 3).await, 3);
    provider.set_unavailable(false);
    // A failed fill disables query reads, including any old provider values.
    assert_eq!(query(&cache, "/docs", 4).await, 4);
}

fn filesystem(mode: CacheTraversalMode) -> CachedFileSystem {
    CachedFileSystem::with_runtime(
        Box::new(MemFileSystem::new()),
        CacheRuntime::memory(),
        CacheNamespace::new("integration"),
        CachePolicy::default().with_traversal_mode(mode),
    )
}

async fn seed(fs: &CachedFileSystem) {
    fs.mkdir("/docs", 0o755).await.unwrap();
    fs.mkdir("/docs/sub", 0o755).await.unwrap();
    fs.write(
        "/docs/a.txt",
        b"before\nneedle\nafter\n",
        0,
        WriteFlag::Create,
    )
    .await
    .unwrap();
    fs.write("/docs/sub/b.txt", b"NEEDLE\n", 0, WriteFlag::Create)
        .await
        .unwrap();
}

#[tokio::test]
async fn query_apis_serve_cached_results_and_parameter_variants() {
    let fs = filesystem(CacheTraversalMode::Backend);
    seed(&fs).await;
    let stat = fs.stat("/docs/a.txt").await.unwrap();
    let dirs = fs
        .read_dir(
            "/docs",
            None,
            None,
            Some(ListSortBy::Name),
            Some(SortOrder::Asc),
        )
        .await
        .unwrap();
    assert_eq!(dirs.len(), 2);
    let page = fs
        .read_dir(
            "/docs",
            Some(1),
            Some(1),
            Some(ListSortBy::Name),
            Some(SortOrder::Asc),
        )
        .await
        .unwrap();
    assert_eq!(page.len(), 1);
    assert_eq!(page[0].name, dirs[1].name);
    let tree = fs
        .tree_directory("/docs", false, None, None, None, None, None, false)
        .await
        .unwrap();
    let only_dirs = fs
        .tree_directory("/docs", false, None, None, None, None, None, true)
        .await
        .unwrap();
    assert!(tree.len() > only_dirs.len());
    assert!(only_dirs.iter().all(|entry| entry.info.is_dir));
    let glob = fs
        .glob_directory("/docs", "**/*.txt", false, None, None, None)
        .await
        .unwrap();
    let unmatched = fs
        .glob_directory("/docs", "**/*.md", false, None, None, None)
        .await
        .unwrap();
    assert!(!glob.entries.is_empty());
    assert!(unmatched.entries.is_empty());
    let options = GrepOptions {
        recursive: true,
        ..Default::default()
    };
    let grep = fs.grep("/docs", "needle", options).await.unwrap();
    let insensitive = fs
        .grep(
            "/docs",
            "needle",
            GrepOptions {
                case_insensitive: true,
                ..options
            },
        )
        .await
        .unwrap();
    assert!(insensitive.matches.len() > grep.matches.len());
    let context = fs
        .grep(
            "/docs",
            "needle",
            GrepOptions {
                before_context: 1,
                after_context: 1,
                ..options
            },
        )
        .await
        .unwrap();
    assert_eq!(context.matches[0].before_context.as_ref().unwrap().len(), 1);
    assert_eq!(context.matches[0].after_context.as_ref().unwrap().len(), 1);

    // Removing the backend data directly makes any unexpected miss observable.
    fs.inner_fs().remove_all("/docs").await.unwrap();
    assert_eq!(fs.stat("/docs/a.txt").await.unwrap().size, stat.size);
    assert_eq!(
        fs.read_dir(
            "/docs",
            None,
            None,
            Some(ListSortBy::Name),
            Some(SortOrder::Asc)
        )
        .await
        .unwrap()
        .len(),
        dirs.len()
    );
    assert_eq!(
        fs.tree_directory("/docs", false, None, None, None, None, None, false)
            .await
            .unwrap()
            .len(),
        tree.len()
    );
    assert_eq!(
        fs.glob_directory("/docs", "**/*.txt", false, None, None, None)
            .await
            .unwrap()
            .entries
            .len(),
        glob.entries.len()
    );
    assert_eq!(
        fs.grep("/docs", "needle", options)
            .await
            .unwrap()
            .matches
            .len(),
        grep.matches.len()
    );
}

#[tokio::test]
async fn writes_rename_and_recursive_remove_refresh_queries_in_both_traversal_modes() {
    for mode in [
        CacheTraversalMode::Backend,
        CacheTraversalMode::CachedTraversal,
    ] {
        let fs = filesystem(mode);
        seed(&fs).await;
        fs.stat("/docs/a.txt").await.unwrap();
        fs.stat("/docs/sub/b.txt").await.unwrap();
        fs.read_dir("/docs", None, None, None, None).await.unwrap();
        fs.tree_directory("/docs", false, None, None, None, None, None, false)
            .await
            .unwrap();
        fs.glob_directory("/docs", "**/*.txt", false, None, None, None)
            .await
            .unwrap();
        let options = GrepOptions {
            recursive: true,
            ..Default::default()
        };
        fs.grep("/docs", "needle", options).await.unwrap();
        fs.write("/docs/a.txt", b"changed", 0, WriteFlag::Truncate)
            .await
            .unwrap();
        assert_eq!(fs.stat("/docs/a.txt").await.unwrap().size, 7);
        assert!(fs
            .grep("/docs", "needle", options)
            .await
            .unwrap()
            .matches
            .is_empty());
        fs.rename("/docs/sub", "/docs/moved").await.unwrap();
        assert!(fs.stat("/docs/sub/b.txt").await.is_err());
        assert!(fs.stat("/docs/moved/b.txt").await.is_ok());
        let tree = fs
            .tree_directory("/docs", false, None, None, None, None, None, false)
            .await
            .unwrap();
        assert!(tree.iter().any(|entry| entry.path.contains("moved")));
        assert!(!tree.iter().any(|entry| entry.path.contains("/sub")));
        fs.remove_all("/docs/moved").await.unwrap();
        assert!(fs.stat("/docs/moved/b.txt").await.is_err());
        assert_eq!(
            fs.read_dir("/docs", None, None, None, None)
                .await
                .unwrap()
                .len(),
            1
        );
        assert_eq!(
            fs.glob_directory("/docs", "**/*.txt", false, None, None, None)
                .await
                .unwrap()
                .entries
                .len(),
            1
        );
    }
}

#[tokio::test]
async fn internal_directory_scans_bypass_cached_public_listing() {
    let fs = filesystem(CacheTraversalMode::Backend);
    seed(&fs).await;
    fs.read_dir("/docs", None, None, None, None).await.unwrap();
    fs.inner_fs().create("/docs/new.txt").await.unwrap();
    assert_eq!(fs.read_internal_dir("/docs").await.unwrap().len(), 3);
}
