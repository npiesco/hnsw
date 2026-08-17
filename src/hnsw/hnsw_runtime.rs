use super::feature_store::FeatureStore;
use super::nodes::Layer;
use crate::*;
use alloc::{vec, vec::Vec};
use core::marker::PhantomData;
use rand_core::{RngCore, SeedableRng};
use space::{Metric, Neighbor};

/// Sentinel for an empty neighbor slot (mirrors the const-generic `!0` fill).
const EMPTY: usize = !0;

/// A non-zero-layer node in the runtime-degree HNSW.
///
/// Structurally identical to the const-generic [`crate::Node`] except the
/// neighbor list is a heap `Vec<usize>` of runtime length `m` (fixed at
/// construction, `EMPTY`-padded) instead of a `[usize; M]` array.
#[derive(Clone, Debug)]
struct RuntimeNode {
    zero_node: usize,
    next_node: usize,
    /// Length == `m`, `EMPTY`-padded. Live neighbors are the leading run of
    /// non-`EMPTY` entries.
    neighbors: Vec<usize>,
}

/// Live neighbors = the leading run of non-`EMPTY` entries (mirrors
/// `NeighborNodes::get_neighbors`'s `take_while(|&n| n != !0)`).
fn live_neighbors(slice: &[usize]) -> impl Iterator<Item = usize> + '_ {
    slice.iter().copied().take_while(|&n| n != EMPTY)
}

/// Runtime-degree HNSW: behaviorally identical to
/// [`crate::Hnsw`]`<Met, T, R, M, M0>` but with the graph out-degrees `M`
/// (upper layers) and `M0` (zero layer) chosen at RUNTIME rather than as
/// compile-time const generics.
///
/// # Why this exists
///
/// The const-generic [`crate::Hnsw`] bakes `M`/`M0` into the type, which is the
/// right call for a persisted index (the on-disk snapshot format embeds the
/// concrete type). But an operator-configured, transient, non-persisted index
/// cannot monomorphize over an arbitrary runtime `M` without an enum-dispatch
/// whitelist of fixed values. This type stores the degrees as `usize` fields
/// and sizes its neighbor `Vec`s accordingly, so any `m >= 2` (`m0 >= m`) is
/// supported with no fixed set of allowed values.
///
/// Given the SAME `(m, m0)`, the SAME seeded PRNG, and the SAME insertion order,
/// this produces byte-identical construction + search results to
/// `Hnsw<_, _, _, M, M0>`; see `tests/runtime_parity.rs`.
#[derive(Clone)]
pub struct HnswRuntime<Met, T, R, S = Vec<T>> {
    metric: Met,
    /// Upper-layer out-degree (the const-generic `M`).
    m: usize,
    /// Zero-layer out-degree (the const-generic `M0`).
    m0: usize,
    /// Zero-layer neighbor lists. Each entry has length `m0`, `EMPTY`-padded.
    zero: Vec<Vec<usize>>,
    /// Zero-layer features, indexed by zero-node id.
    features: S,
    /// Non-zero layers.
    layers: Vec<Vec<RuntimeNode>>,
    prng: R,
    params: Params,
    /// Soft-delete tombstones indexed by zero-node id: `deleted[id] == true`
    /// means node `id` is logically removed. Mirrors the const-generic index's
    /// tombstone state. Grown lazily; reads past its end are live.
    deleted: Vec<bool>,
    /// `T` appears only behind the storage backend, so it needs anchoring here.
    _marker: PhantomData<T>,
}

impl<Met, T, R> HnswRuntime<Met, T, R, Vec<T>>
where
    R: RngCore + SeedableRng,
{
    /// Creates a runtime-degree HNSW with a default-seeded deterministic PRNG.
    ///
    /// # Panics
    /// Panics if `m < 2` or `m0 < m` (checked topology guardrails, not tuning
    /// defaults): `m < 2` cannot form a navigable graph and the level
    /// distribution `1/ln(m)` is undefined at `m == 1`; the zero layer must be
    /// at least as connected as the upper layers.
    pub fn new(metric: Met, m: usize, m0: usize) -> Self {
        Self::new_params_and_prng(
            metric,
            m,
            m0,
            Params::new(),
            R::from_seed(R::Seed::default()),
        )
    }

    /// Creates a runtime-degree HNSW with the specified params and a
    /// default-seeded PRNG.
    pub fn new_params(metric: Met, m: usize, m0: usize, params: Params) -> Self {
        Self::new_params_and_prng(metric, m, m0, params, R::from_seed(R::Seed::default()))
    }
}

impl<Met, T, R> HnswRuntime<Met, T, R, Vec<T>>
where
    R: RngCore,
{
    /// Creates a runtime-degree HNSW with the passed `prng`.
    pub fn new_prng(metric: Met, m: usize, m0: usize, prng: R) -> Self {
        Self::new_params_and_prng(metric, m, m0, Params::default(), prng)
    }

    /// Creates a runtime-degree HNSW with the passed `params` and `prng`.
    ///
    /// # Panics
    /// See [`Self::new`] for the `m`/`m0` guardrails.
    pub fn new_params_and_prng(metric: Met, m: usize, m0: usize, params: Params, prng: R) -> Self {
        Self::new_with_storage_and_params(metric, m, m0, vec![], params, prng)
    }
}

impl<Met, T, R, S> HnswRuntime<Met, T, R, S>
where
    R: RngCore,
    S: FeatureStore<T>,
{
    /// Creates a runtime-degree HNSW over a custom feature storage backend.
    ///
    /// See [`FeatureStore`] for the contract a backend must uphold.
    pub fn new_with_storage(metric: Met, m: usize, m0: usize, storage: S, prng: R) -> Self {
        Self::new_with_storage_and_params(metric, m, m0, storage, Params::default(), prng)
    }

    /// Creates a runtime-degree HNSW over a custom feature storage backend with
    /// the passed `params`.
    ///
    /// See [`FeatureStore`] for the contract a backend must uphold.
    ///
    /// # Panics
    /// See [`Self::new`] for the `m`/`m0` guardrails.
    pub fn new_with_storage_and_params(
        metric: Met,
        m: usize,
        m0: usize,
        storage: S,
        params: Params,
        prng: R,
    ) -> Self {
        assert!(m >= 2, "HnswRuntime requires m >= 2, got {}", m);
        assert!(
            m0 >= m,
            "HnswRuntime requires m0 >= m, got m0={} m={}",
            m0,
            m
        );
        Self {
            metric,
            m,
            m0,
            zero: vec![],
            features: storage,
            layers: vec![],
            prng,
            params,
            deleted: vec![],
            _marker: PhantomData,
        }
    }

    /// Upper-layer out-degree (`M`).
    pub fn m(&self) -> usize {
        self.m
    }

    /// Zero-layer out-degree (`M0`).
    pub fn m0(&self) -> usize {
        self.m0
    }

    pub fn len(&self) -> usize {
        self.zero.len()
    }

    pub fn is_empty(&self) -> bool {
        self.zero.is_empty()
    }

    pub fn layers(&self) -> usize {
        self.layers.len() + 1
    }

    /// Extract the feature for a given item returned by [`Self::nearest`].
    pub fn feature(&self, item: usize) -> &T {
        self.features.get_feature(item)
    }

    /// Number of live (non-deleted) nodes in the index.
    pub fn live_count(&self) -> usize {
        self.zero.len() - self.deleted.iter().filter(|&&d| d).count()
    }

    /// Returns whether zero-node `id` has been soft-deleted.
    #[inline]
    pub fn is_deleted(&self, id: usize) -> bool {
        self.deleted.get(id).copied().unwrap_or(false)
    }

    /// Soft-deletes zero-node `id`: it is excluded from search results but its
    /// slot and edges remain in the graph so navigation stays connected until an
    /// exact rebuild (compaction) reclaims it. Idempotent.
    pub fn mark_delete(&mut self, id: usize) {
        if id >= self.deleted.len() {
            self.deleted.resize(id + 1, false);
        }
        self.deleted[id] = true;
    }

    /// The zero-node id of the current live navigation entry point, or `None`
    /// if the index has no live nodes. Scans the towers top-down and returns the
    /// first live node, so a deleted entry point is transparently skipped.
    pub fn entry(&self) -> Option<usize> {
        if self.zero.is_empty() {
            return None;
        }
        for layer in self.layers.iter().rev() {
            for node in layer {
                if !self.is_deleted(node.zero_node) {
                    return Some(node.zero_node);
                }
            }
        }
        (0..self.zero.len()).find(|&id| !self.is_deleted(id))
    }
}

impl<Met, T, R, S> HnswRuntime<Met, T, R, S>
where
    R: RngCore,
    Met: Metric<T>,
    S: FeatureStore<T>,
{
    /// Inserts a feature into the HNSW. Mirrors [`crate::Hnsw::insert`].
    pub fn insert(&mut self, q: T, searcher: &mut Searcher<Met::Unit>) -> usize {
        let level = self.random_level();
        let mut cap = if level >= self.layers.len() {
            self.params.ef_construction
        } else {
            1
        };

        if self.is_empty() {
            self.zero.push(vec![EMPTY; self.m0]);
            self.features.push_feature(q);
            while self.layers.len() < level {
                let node = RuntimeNode {
                    zero_node: 0,
                    next_node: 0,
                    neighbors: vec![EMPTY; self.m],
                };
                self.layers.push(vec![node]);
            }
            return 0;
        }

        self.initialize_searcher(&q, searcher);

        for ix in (level..self.layers.len()).rev() {
            self.search_single_layer(&q, searcher, Layer::NonZero(&self.layers[ix]), cap);
            Self::lower_search(&self.layers[ix], searcher);
            cap = if ix == level {
                self.params.ef_construction
            } else {
                1
            };
        }

        for ix in (0..core::cmp::min(level, self.layers.len())).rev() {
            self.search_single_layer(&q, searcher, Layer::NonZero(&self.layers[ix]), cap);
            self.create_node(&q, &searcher.nearest, ix + 1);
            Self::lower_search(&self.layers[ix], searcher);
            cap = self.params.ef_construction;
        }

        self.search_zero_layer(&q, searcher, cap);
        self.create_node(&q, &searcher.nearest, 0);
        self.features.push_feature(q);

        let zero_node = self.zero.len() - 1;
        while self.layers.len() < level {
            let node = RuntimeNode {
                zero_node,
                next_node: self.layers.last().map(|l| l.len() - 1).unwrap_or(zero_node),
                neighbors: vec![EMPTY; self.m],
            };
            self.layers.push(vec![node]);
        }
        zero_node
    }

    /// k-NN search. Mirrors [`crate::Hnsw::nearest`].
    pub fn nearest<'a>(
        &self,
        q: &T,
        ef: usize,
        searcher: &mut Searcher<Met::Unit>,
        dest: &'a mut [Neighbor<Met::Unit>],
    ) -> &'a mut [Neighbor<Met::Unit>] {
        self.search_layer(q, ef, 0, searcher, dest)
    }

    /// Mirrors [`crate::Hnsw::search_layer`].
    pub fn search_layer<'a>(
        &self,
        q: &T,
        ef: usize,
        level: usize,
        searcher: &mut Searcher<Met::Unit>,
        dest: &'a mut [Neighbor<Met::Unit>],
    ) -> &'a mut [Neighbor<Met::Unit>] {
        if self.features.feature_count() == 0 || level >= self.layers() {
            return &mut [];
        }

        self.initialize_searcher(q, searcher);
        let cap = 1;

        for (ix, layer) in self.layers.iter().enumerate().rev() {
            self.search_single_layer(q, searcher, Layer::NonZero(layer), cap);
            if ix + 1 == level {
                let found = core::cmp::min(dest.len(), searcher.nearest.len());
                dest[..found].copy_from_slice(&searcher.nearest[..found]);
                return &mut dest[..found];
            }
            Self::lower_search(layer, searcher);
        }

        let cap = ef;
        self.search_zero_layer(q, searcher, cap);

        // The zero-layer search excludes soft-deleted nodes from the result
        // heap, but `lower_search` may have seeded `nearest` with a deleted node
        // when descending from the upper layers. Drop those so a tombstone can
        // never surface as a result.
        searcher.nearest.retain(|n| !self.is_deleted(n.index));

        let found = core::cmp::min(dest.len(), searcher.nearest.len());
        dest[..found].copy_from_slice(&searcher.nearest[..found]);
        &mut dest[..found]
    }

    /// Mirrors [`crate::Hnsw::search_single_layer`].
    fn search_single_layer(
        &self,
        q: &T,
        searcher: &mut Searcher<Met::Unit>,
        layer: Layer<&[RuntimeNode]>,
        cap: usize,
    ) {
        while let Some(Neighbor { index, .. }) = searcher.candidates.pop() {
            let raw_neighbors: &[usize] = match layer {
                Layer::NonZero(layer) => &layer[index].neighbors,
                Layer::Zero => &self.zero[index],
            };
            for neighbor in live_neighbors(raw_neighbors) {
                let node_to_visit = match layer {
                    Layer::NonZero(layer) => layer[neighbor].zero_node,
                    Layer::Zero => neighbor,
                };
                if searcher.seen.insert(node_to_visit) {
                    let distance = self
                        .metric
                        .distance(q, self.features.get_feature(node_to_visit));
                    // At the zero (result) layer a soft-deleted node is still
                    // traversed — so live nodes reachable only through it stay
                    // reachable — but it must NEVER enter the result heap nor
                    // consume the `cap` budget, otherwise the result set
                    // under-fills with live neighbors crowded out by tombstones.
                    if matches!(layer, Layer::Zero) && self.is_deleted(node_to_visit) {
                        searcher.candidates.push(Neighbor {
                            index: neighbor,
                            distance,
                        });
                        continue;
                    }
                    let pos = searcher.nearest.partition_point(|n| n.distance <= distance);
                    if pos != cap {
                        if searcher.nearest.len() == cap {
                            searcher.nearest.pop();
                        }
                        let candidate = Neighbor {
                            index: neighbor,
                            distance,
                        };
                        searcher.nearest.insert(pos, candidate);
                        searcher.candidates.push(candidate);
                    }
                }
            }
        }
    }

    fn search_zero_layer(&self, q: &T, searcher: &mut Searcher<Met::Unit>, cap: usize) {
        self.search_single_layer(q, searcher, Layer::Zero, cap);
    }

    /// Mirrors [`crate::Hnsw::lower_search`].
    fn lower_search(layer: &[RuntimeNode], searcher: &mut Searcher<Met::Unit>) {
        searcher.candidates.clear();
        let &Neighbor { index, distance } = searcher.nearest.first().unwrap();
        searcher.nearest.clear();
        searcher.seen.clear();
        let new_index = layer[index].next_node;
        let candidate = Neighbor {
            index: new_index,
            distance,
        };
        searcher.seen.insert(layer[index].zero_node);
        searcher.nearest.push(candidate);
        searcher.candidates.push(candidate);
    }

    /// Mirrors [`crate::Hnsw::initialize_searcher`].
    fn initialize_searcher(&self, q: &T, searcher: &mut Searcher<Met::Unit>) {
        searcher.clear();
        let entry_distance = self.metric.distance(q, self.entry_feature());
        let candidate = Neighbor {
            index: 0,
            distance: entry_distance,
        };
        searcher.candidates.push(candidate);
        searcher.nearest.push(candidate);
        searcher.seen.insert(
            self.layers
                .last()
                .map(|layer| layer[0].zero_node)
                .unwrap_or(0),
        );
    }

    fn entry_feature(&self) -> &T {
        if let Some(last_layer) = self.layers.last() {
            self.features.get_feature(last_layer[0].zero_node)
        } else {
            self.features.get_feature(0)
        }
    }

    /// Mirrors [`crate::Hnsw::random_level`], using the runtime `m`.
    fn random_level(&mut self) -> usize {
        let uniform: f64 = self.prng.next_u64() as f64 / u64::MAX as f64;
        (-libm::log(uniform) * libm::log(self.m as f64).recip()) as usize
    }

    /// Mirrors [`crate::Hnsw::create_node`].
    fn create_node(&mut self, q: &T, nearest: &[Neighbor<Met::Unit>], layer: usize) {
        if layer == 0 {
            let new_index = self.zero.len();
            let mut neighbors = vec![EMPTY; self.m0];
            for (d, s) in neighbors.iter_mut().zip(nearest.iter()) {
                *d = s.index;
            }
            let live: Vec<usize> = live_neighbors(&neighbors).collect();
            for neighbor in live {
                self.add_neighbor(q, new_index, neighbor, layer);
            }
            self.zero.push(neighbors);
        } else {
            let new_index = self.layers[layer - 1].len();
            let mut neighbors = vec![EMPTY; self.m];
            for (d, s) in neighbors.iter_mut().zip(nearest.iter()) {
                *d = s.index;
            }
            let node = RuntimeNode {
                zero_node: self.zero.len(),
                next_node: if layer == 1 {
                    self.zero.len()
                } else {
                    self.layers[layer - 2].len()
                },
                neighbors,
            };
            let live: Vec<usize> = live_neighbors(&node.neighbors).collect();
            for neighbor in live {
                self.add_neighbor(q, new_index, neighbor, layer);
            }
            self.layers[layer - 1].push(node);
        }
    }

    /// Mirrors [`crate::Hnsw::add_neighbor`], including its neighbor selection
    /// heuristic. The two must stay identical: `tests/runtime_parity.rs` holds
    /// this index byte-identical to the const-generic one for the same degrees,
    /// seed, and insertion order, and neighbor pruning is part of what it
    /// compares.
    fn add_neighbor(&mut self, q: &T, node_ix: usize, target_ix: usize, layer: usize) {
        let capacity = if layer == 0 { self.m0 } else { self.m };

        let existing: Vec<usize> = if layer == 0 {
            self.zero[target_ix]
                .iter()
                .copied()
                .take_while(|&n| n != EMPTY)
                .collect()
        } else {
            self.layers[layer - 1][target_ix]
                .neighbors
                .iter()
                .copied()
                .take_while(|&n| n != EMPTY)
                .collect()
        };

        if existing.len() < capacity {
            let slot = existing.len();
            if layer == 0 {
                self.zero[target_ix][slot] = node_ix;
            } else {
                self.layers[layer - 1][target_ix].neighbors[slot] = node_ix;
            }
            return;
        }

        let kept: Vec<usize> = {
            let feature_of = |ix: usize| -> &T {
                if ix == node_ix {
                    q
                } else if layer == 0 {
                    self.features.get_feature(ix)
                } else {
                    self.features
                        .get_feature(self.layers[layer - 1][ix].zero_node)
                }
            };

            let target_feature = feature_of(target_ix);

            let mut candidates: Vec<(usize, Met::Unit)> = Vec::with_capacity(capacity + 1);
            for &n in &existing {
                candidates.push((n, self.metric.distance(target_feature, feature_of(n))));
            }
            candidates.push((node_ix, self.metric.distance(target_feature, q)));
            candidates.sort_unstable_by_key(|&(_, distance)| distance);

            let mut kept: Vec<usize> = Vec::with_capacity(capacity);
            let mut pruned: Vec<usize> = Vec::with_capacity(capacity);
            for &(ix, distance_to_target) in &candidates {
                if kept.len() == capacity {
                    break;
                }
                let candidate = feature_of(ix);
                let diverse = kept
                    .iter()
                    .all(|&r| distance_to_target < self.metric.distance(candidate, feature_of(r)));
                if diverse {
                    kept.push(ix);
                } else {
                    pruned.push(ix);
                }
            }

            for ix in pruned {
                if kept.len() == capacity {
                    break;
                }
                kept.push(ix);
            }
            kept
        };

        let slots: &mut [usize] = if layer == 0 {
            &mut self.zero[target_ix][..]
        } else {
            &mut self.layers[layer - 1][target_ix].neighbors[..]
        };
        for (slot, value) in slots
            .iter_mut()
            .zip(kept.into_iter().chain(core::iter::repeat(EMPTY)))
        {
            *slot = value;
        }
    }
}

impl<Met, T, R> Default for HnswRuntime<Met, T, R>
where
    R: RngCore + SeedableRng,
    Met: Default,
{
    /// Default runtime degrees mirror the crate's canonical const-generic
    /// instantiation (`M = 12`, `M0 = 24`). Callers that need operator-chosen
    /// degrees use [`Self::new`] / [`Self::new_params_and_prng`].
    fn default() -> Self {
        Self::new(Met::default(), 12, 24)
    }
}
