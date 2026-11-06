// Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
// SPDX-License-Identifier: AGPL-3.0

//! Result buckets kept separate from file bodies and raw directory entries.

use super::metrics::CacheMetrics;
use super::wrapper::{ancestor_scopes, normalize_path, stable_hash};
use crate::cache_runtime::{CacheRuntime, Expiration, SetOptions, SetResult};
use crate::core::Result;
use bytes::Bytes;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::future::Future;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

#[derive(Default)]
struct State {
    revision: u64,
    paths: BTreeSet<String>,
    disabled: bool,
}

#[derive(Serialize, Deserialize)]
struct Bucket {
    path: String,
    queries: BTreeMap<String, Value>,
}

pub(super) struct QueryCache {
    prefix: String,
    runtime: Arc<CacheRuntime>,
    metrics: Arc<CacheMetrics>,
    state: RwLock<State>,
}

impl QueryCache {
    pub(super) fn new(
        namespace: &str,
        epoch: u64,
        runtime: Arc<CacheRuntime>,
        metrics: Arc<CacheMetrics>,
    ) -> Self {
        Self {
            // The path index belongs to this wrapper. A new wrapper must never
            // read buckets whose invalidation index was lost with its owner.
            prefix: format!("ragfs:v1:{namespace}:query:{epoch:016x}"),
            runtime,
            metrics,
            state: RwLock::new(State::default()),
        }
    }

    fn key(&self, path: &str) -> String {
        format!("{}:{:016x}", self.prefix, stable_hash(path.as_bytes()))
    }

    async fn bucket(&self, path: &str) -> Option<Bucket> {
        let started = Instant::now();
        let result = self.runtime.get(&self.key(path)).await;
        self.metrics.get(started.elapsed());
        match result {
            Ok(Some(bytes)) => match serde_json::from_slice::<Bucket>(&bytes) {
                Ok(bucket) if bucket.path == path => Some(bucket),
                _ => {
                    self.metrics.error();
                    None
                }
            },
            Ok(None) => None,
            Err(_) => {
                self.metrics.error();
                None
            }
        }
    }

    pub(super) async fn get_or_load<T, F>(
        &self,
        path: &str,
        op: &str,
        params: Value,
        load: F,
    ) -> Result<T>
    where
        T: Serialize + DeserializeOwned + Send + Sync,
        F: Future<Output = Result<T>> + Send,
    {
        let path = normalize_path(path);
        // Store the complete request as the field, avoiding query-hash collisions.
        let request = serde_json::json!([op, params]).to_string();
        let revision = {
            let state = self.state.read().await;
            if state.disabled {
                drop(state);
                return load.await;
            }
            if let Some(bucket) = self.bucket(&path).await {
                if let Some(value) = bucket.queries.get(&request) {
                    if let Ok(payload) = serde_json::from_value(value.clone()) {
                        return Ok(payload);
                    }
                    self.metrics.error();
                }
            }
            state.revision
        };

        // Never hold the query lock across a backend/traversal call: cached
        // traversal can recursively issue read_dir queries through this wrapper.
        let result = load.await;
        if let Ok(payload) = &result {
            let mut state = self.state.write().await;
            if !state.disabled && state.revision == revision {
                state.paths.insert(path.clone());
                if let Ok(value) = serde_json::to_value(payload) {
                    let mut bucket = self.bucket(&path).await.unwrap_or_else(|| Bucket {
                        path: path.clone(),
                        queries: BTreeMap::new(),
                    });
                    bucket.queries.insert(request, value);
                    if let Ok(bytes) = serde_json::to_vec(&bucket) {
                        let started = Instant::now();
                        let stored = self
                            .runtime
                            .set(
                                &self.key(&path),
                                Bytes::from(bytes),
                                SetOptions {
                                    // Reclaim buckets after their owning wrapper exits.
                                    expiration: Some(Expiration::After(Duration::from_secs(300))),
                                    ..SetOptions::default()
                                },
                            )
                            .await;
                        self.metrics.put(started.elapsed());
                        if !matches!(stored, Ok(SetResult::Applied)) {
                            self.metrics.error();
                            state.disabled = true;
                        }
                    }
                }
            }
        }
        result
    }

    pub(super) async fn invalidate(&self, path: &str) {
        let path = normalize_path(path);
        let mut state = self.state.write().await;
        state.revision = state.revision.wrapping_add(1);
        let mut affected = ancestor_scopes(&path);
        let prefix = if path == "/" {
            path
        } else {
            format!("{path}/")
        };
        affected.extend(
            state
                .paths
                .range(prefix.clone()..)
                .take_while(|candidate| candidate.starts_with(&prefix))
                .cloned(),
        );
        let keys: Vec<_> = affected
            .into_iter()
            .filter(|scope| state.paths.remove(scope))
            .map(|scope| self.key(&scope))
            .collect();
        if keys.is_empty() {
            return;
        }
        let started = Instant::now();
        let result = self.runtime.del(&keys).await;
        self.metrics.delete(started.elapsed());
        if result.is_err() {
            // Failed deletion must not expose stale results. Body caching has
            // its own failure handling and is deliberately unaffected here.
            self.metrics.error();
            state.disabled = true;
        } else {
            self.metrics.invalidation();
        }
    }
}

#[cfg(test)]
mod tests;
