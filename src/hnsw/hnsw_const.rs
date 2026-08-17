use super::feature_store::FeatureStore;
use super::nodes::{HasNeighbors, Layer};
use crate::hnsw::nodes::{NeighborNodes, Node};
use crate::*;
use alloc::{vec, vec::Vec};
use core::marker::PhantomData;
use num_traits::Zero;
use rand_core::{RngCore, SeedableRng};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use space::{Knn, KnnPoints, Metric, Neighbor};

/// This provides a HNSW implementation for any distance function.
///
/// The type `T` must implement [`space::Metric`] to get implementations.
///
/// The type `S` is the feature storage backend and defaults to [`Vec<T>`].
/// Supplying another [`FeatureStore`] keeps the features outside the heap — in
/// an `mmap`ed file, say — while the graph stays in memory.
#[derive(Clone)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(bound(
        serialize = "Met: Serialize, T: Serialize, R: Serialize, S: Serialize",
        deserialize = "Met: Deserialize<'de>, T: Deserialize<'de>, R: Deserialize<'de>, S: Deserialize<'de>"
    ))
)]
pub struct Hnsw<Met, T, R, const M: usize, const M0: usize, S = Vec<T>> {
    /// Contains the space metric.
    metric: Met,
    /// Contains the zero layer.
    zero: Vec<NeighborNodes<M0>>,
    /// Contains the features of the zero layer.
    /// These are stored separately to allow SIMD speedup in the future by
    /// grouping small worlds of features together.
    features: S,
    /// Contains each non-zero layer.
    layers: Vec<Vec<Node<M>>>,
    /// This needs to create resonably random outputs to determine the levels of insertions.
    prng: R,
    /// The parameters for the HNSW.
    params: Params,
    /// Soft-delete tombstones indexed by zero-node id: `deleted[id] == true`
    /// means node `id` is logically removed. This is DERIVED state — it is
    /// `#[serde(skip)]` so the on-disk snapshot format is unchanged, and the
    /// owner (immutlex) re-applies tombstones on load from its id maps via
    /// [`Hnsw::mark_delete`]. Grown lazily; reads past its end are live.
    #[cfg_attr(feature = "serde", serde(skip))]
    deleted: Vec<bool>,
    /// `T` appears only behind the storage backend, so it needs anchoring here.
    /// Skipped by serde so the snapshot format is unchanged.
    #[cfg_attr(feature = "serde", serde(skip))]
    _marker: PhantomData<T>,
}

impl<Met, T, R, const M: usize, const M0: usize> Hnsw<Met, T, R, M, M0, Vec<T>>
where
    R: RngCore + SeedableRng,
{
    /// Creates a new HNSW with a PRNG which is default seeded to produce deterministic behavior.
    pub fn new(metric: Met) -> Self {
        Self {
            metric,
            zero: vec![],
            features: vec![],
            layers: vec![],
            prng: R::from_seed(R::Seed::default()),
            params: Params::new(),
            deleted: vec![],
            _marker: PhantomData,
        }
    }

    /// Creates a new HNSW with a default seeded PRNG and with the specified params.
    pub fn new_params(metric: Met, params: Params) -> Self {
        Self {
            metric,
            zero: vec![],
            features: vec![],
            layers: vec![],
            prng: R::from_seed(R::Seed::default()),
            params,
            deleted: vec![],
            _marker: PhantomData,
        }
    }

    pub fn new_with_capacity(metric: Met, params: Params, capacity: usize) -> Self {
        Self {
            metric,
            zero: Vec::with_capacity(capacity),
            features: Vec::with_capacity(capacity),
            layers: vec![],
            prng: R::from_seed(R::Seed::default()),
            params,
            deleted: vec![],
            _marker: PhantomData,
        }
    }
}

impl<Met, T, R, const M: usize, const M0: usize, S> Knn for Hnsw<Met, T, R, M, M0, S>
where
    R: RngCore,
    Met: Metric<T>,
    S: FeatureStore<T>,
{
    type Ix = usize;
    type Metric = Met;
    type Point = T;
    type KnnIter = Vec<Neighbor<Met::Unit>>;

    fn knn(&self, query: &T, num: usize) -> Self::KnnIter {
        let mut searcher = Searcher::default();
        let mut neighbors = vec![
            Neighbor {
                index: !0,
                distance: Met::Unit::zero(),
            };
            num
        ];
        let found = self
            .nearest(query, num + 16, &mut searcher, &mut neighbors)
            .len();
        neighbors.resize_with(found, || unreachable!());
        neighbors
    }
}

impl<Met, T, R, const M: usize, const M0: usize, S> KnnPoints for Hnsw<Met, T, R, M, M0, S>
where
    R: RngCore,
    Met: Metric<T>,
    S: FeatureStore<T>,
{
    fn get_point(&self, index: usize) -> &'_ T {
        self.features.get_feature(index)
    }
}

impl<Met, T, R, const M: usize, const M0: usize> Hnsw<Met, T, R, M, M0, Vec<T>>
where
    R: RngCore,
    Met: Metric<T>,
{
    /// Creates a HNSW with the passed `prng`.
    pub fn new_prng(metric: Met, prng: R) -> Self {
        Self {
            metric,
            zero: vec![],
            features: vec![],
            layers: vec![],
            prng,
            params: Default::default(),
            deleted: vec![],
            _marker: PhantomData,
        }
    }

    /// Creates a HNSW with the passed `params` and `prng`.
    pub fn new_params_and_prng(metric: Met, params: Params, prng: R) -> Self {
        Self {
            metric,
            zero: vec![],
            features: vec![],
            layers: vec![],
            prng,
            params,
            deleted: vec![],
            _marker: PhantomData,
        }
    }
}

impl<Met, T, R, const M: usize, const M0: usize, S> Hnsw<Met, T, R, M, M0, S>
where
    R: RngCore,
    Met: Metric<T>,
    S: FeatureStore<T>,
{
    /// Creates a HNSW over a custom feature storage backend.
    ///
    /// See [`FeatureStore`] for the contract a backend must uphold.
    pub fn new_with_storage(metric: Met, storage: S, prng: R) -> Self {
        Self {
            metric,
            zero: vec![],
            features: storage,
            layers: vec![],
            prng,
            params: Default::default(),
            deleted: vec![],
            _marker: PhantomData,
        }
    }

    /// Creates a HNSW over a custom feature storage backend with the passed
    /// `params`.
    ///
    /// See [`FeatureStore`] for the contract a backend must uphold.
    pub fn new_with_storage_and_params(metric: Met, storage: S, params: Params, prng: R) -> Self {
        Self {
            metric,
            zero: vec![],
            features: storage,
            layers: vec![],
            prng,
            params,
            deleted: vec![],
            _marker: PhantomData,
        }
    }

    /// Inserts a feature into the HNSW.
    pub fn insert(&mut self, q: T, searcher: &mut Searcher<Met::Unit>) -> usize {
        // Get the level of this feature.
        let level = self.random_level();
        let mut cap = if level >= self.layers.len() {
            self.params.ef_construction
        } else {
            1
        };

        // If this is empty, none of this will work, so just add it manually.
        if self.is_empty() {
            // Add the zero node unconditionally.
            self.zero.push(NeighborNodes {
                neighbors: [!0; M0],
            });
            self.features.push_feature(q);

            // Add all the layers its in.
            while self.layers.len() < level {
                // It's always index 0 with no neighbors since its the first feature.
                let node = Node {
                    zero_node: 0,
                    next_node: 0,
                    neighbors: NeighborNodes { neighbors: [!0; M] },
                };
                self.layers.push(vec![node]);
            }
            return 0;
        }

        self.initialize_searcher(&q, searcher);

        // Find the entry point on the level it was created by searching normally until its level.
        for ix in (level..self.layers.len()).rev() {
            // Perform an ANN search on this layer like normal.
            self.search_single_layer(&q, searcher, Layer::NonZero(&self.layers[ix]), cap);
            // Then lower the search only after we create the node.
            self.lower_search(&self.layers[ix], searcher);
            cap = if ix == level {
                self.params.ef_construction
            } else {
                1
            };
        }

        // Then start from its level and connect it to its nearest neighbors.
        for ix in (0..core::cmp::min(level, self.layers.len())).rev() {
            // Perform an ANN search on this layer like normal.
            self.search_single_layer(&q, searcher, Layer::NonZero(&self.layers[ix]), cap);
            // Then use the results of that search on this layer to connect the nodes.
            self.create_node(&q, &searcher.nearest, ix + 1);
            // Then lower the search only after we create the node.
            self.lower_search(&self.layers[ix], searcher);
            cap = self.params.ef_construction;
        }

        // Also search and connect the node to the zero layer.
        self.search_zero_layer(&q, searcher, cap);
        self.create_node(&q, &searcher.nearest, 0);
        // Add the feature to the zero layer.
        self.features.push_feature(q);

        // Add all level vectors needed to be able to add this level.
        let zero_node = self.zero.len() - 1;
        while self.layers.len() < level {
            let node = Node {
                zero_node,
                next_node: self.layers.last().map(|l| l.len() - 1).unwrap_or(zero_node),
                neighbors: NeighborNodes { neighbors: [!0; M] },
            };
            self.layers.push(vec![node]);
        }
        zero_node
    }

    /// Does a k-NN search where `q` is the query element and it attempts to put up to `M` nearest neighbors into `dest`.
    /// `ef` is the candidate pool size. `ef` can be increased to get better recall at the expense of speed.
    /// If `ef` is less than `dest.len()` then `dest` will only be filled with `ef` elements.
    ///
    /// Returns a slice of the filled neighbors.
    pub fn nearest<'a>(
        &self,
        q: &T,
        ef: usize,
        searcher: &mut Searcher<Met::Unit>,
        dest: &'a mut [Neighbor<Met::Unit>],
    ) -> &'a mut [Neighbor<Met::Unit>] {
        self.search_layer(q, ef, 0, searcher, dest)
    }

    /// Extract the feature for a given item returned by [`HNSW::nearest`].
    ///
    /// The `item` must be retrieved from [`HNSW::search_layer`].
    pub fn feature(&self, item: usize) -> &T {
        self.features.get_feature(item)
    }

    /// Extract the feature from a particular level for a given item returned by [`HNSW::search_layer`].
    pub fn layer_feature(&self, level: usize, item: usize) -> &T {
        self.features.get_feature(self.layer_item_id(level, item))
    }

    /// Retrieve the item ID for a given layer item returned by [`HNSW::search_layer`].
    ///
    /// `level` follows the same convention as [`HNSW::search_layer`] and
    /// [`HNSW::layer_len`]: level `n > 0` is `self.layers[n - 1]`, because level
    /// `0` is the zero layer, which is not stored in `self.layers` at all.
    /// Indexing `self.layers[level]` here instead panicked for every non-zero
    /// level — the item index belongs to the layer below the one being indexed,
    /// and at the top level `self.layers[level]` is out of bounds outright.
    pub fn layer_item_id(&self, level: usize, item: usize) -> usize {
        if level == 0 {
            item
        } else {
            self.layers[level - 1][item].zero_node
        }
    }

    pub fn layers(&self) -> usize {
        self.layers.len() + 1
    }

    pub fn len(&self) -> usize {
        self.zero.len()
    }

    pub fn layer_len(&self, level: usize) -> usize {
        if level == 0 {
            self.features.feature_count()
        } else if level < self.layers() {
            self.layers[level - 1].len()
        } else {
            0
        }
    }

    pub fn is_empty(&self) -> bool {
        self.zero.is_empty()
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

    pub fn layer_is_empty(&self, level: usize) -> bool {
        self.layer_len(level) == 0
    }

    /// Performs the same algorithm as [`HNSW::nearest`], but stops on a particular layer of the network
    /// and returns the unique index on that layer rather than the item index.
    ///
    /// If this is passed a `level` of `0`, then this has the exact same functionality as [`HNSW::nearest`]
    /// since the unique indices at layer `0` are the item indices.
    pub fn search_layer<'a>(
        &self,
        q: &T,
        ef: usize,
        level: usize,
        searcher: &mut Searcher<Met::Unit>,
        dest: &'a mut [Neighbor<Met::Unit>],
    ) -> &'a mut [Neighbor<Met::Unit>] {
        // If there is nothing in here, then just return nothing.
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
            self.lower_search(layer, searcher);
        }

        let cap = ef;

        // `initialize_searcher` seeds `nearest` with the entry point and
        // `lower_search` re-seeds it at every descent; neither checks liveness.
        // Scrub tombstones BEFORE the zero-layer search so they cannot occupy
        // one of the `cap` result slots while it runs — the bounded heap treats
        // itself as full at `nearest.len() == cap` and evicts the worst, so a
        // dead seed crowds out a live neighbor and the result set under-fills by
        // exactly one. `candidates` is deliberately left untouched so live nodes
        // reachable only through a deleted node stay reachable.
        searcher.nearest.retain(|n| !self.is_deleted(n.index));

        // search the zero layer
        self.search_zero_layer(q, searcher, cap);

        let found = core::cmp::min(dest.len(), searcher.nearest.len());
        dest[..found].copy_from_slice(&searcher.nearest[..found]);
        &mut dest[..found]
    }

    /// Greedily finds the approximate nearest neighbors to `q` in a non-zero layer.
    /// This corresponds to Algorithm 2 in the paper.
    fn search_single_layer(
        &self,
        q: &T,
        searcher: &mut Searcher<Met::Unit>,
        layer: Layer<&[Node<M>]>,
        cap: usize,
    ) {
        while let Some(Neighbor { index, .. }) = searcher.candidates.pop() {
            for neighbor in match layer {
                Layer::NonZero(layer) => layer[index].get_neighbors(),
                Layer::Zero => self.zero[index].get_neighbors(),
            } {
                let node_to_visit = match layer {
                    Layer::NonZero(layer) => layer[neighbor].zero_node,
                    Layer::Zero => neighbor,
                };

                // Don't visit previously visited things. We use the zero node to allow reusing the seen filter
                // across all layers since zero nodes are consistent among all layers.
                // TODO: Use Cuckoo Filter or Bloom Filter to speed this up/take less memory.
                if searcher.seen.insert(node_to_visit) {
                    // Compute the distance of this neighbor.
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
                    // Attempt to insert into nearest queue.
                    let pos = searcher.nearest.partition_point(|n| n.distance <= distance);
                    if pos != cap {
                        // It was successful. Now we need to know if its full.
                        if searcher.nearest.len() == cap {
                            // In this case remove the worst item.
                            searcher.nearest.pop();
                        }
                        // Either way, add the new item.
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

    /// Greedily finds the approximate nearest neighbors to `q` in the zero layer.
    fn search_zero_layer(&self, q: &T, searcher: &mut Searcher<Met::Unit>, cap: usize) {
        self.search_single_layer(q, searcher, Layer::Zero, cap);
    }

    /// Ready a search for the next level down.
    ///
    /// `m` is the maximum number of nearest neighbors to consider during the search.
    fn lower_search(&self, layer: &[Node<M>], searcher: &mut Searcher<Met::Unit>) {
        // Clear the candidates so we can fill them with the best nodes in the last layer.
        searcher.candidates.clear();
        // Only preserve the best candidate. The original paper's algorithm uses `1` every time.
        // See Algorithm 5 line 5 of the paper. The paper makes no further comment on why `1` was chosen.
        let &Neighbor { index, distance } = searcher.nearest.first().unwrap();
        searcher.nearest.clear();
        searcher.seen.clear();
        // Update the node to the next layer.
        let new_index = layer[index].next_node;
        let candidate = Neighbor {
            index: new_index,
            distance,
        };
        searcher.seen.insert(layer[index].zero_node);
        // Insert the index of the nearest neighbor into the nearest pool for the next layer.
        searcher.nearest.push(candidate);
        // Insert the index into the candidate pool as well.
        searcher.candidates.push(candidate);
    }

    /// Resets a searcher, but does not set the `cap` on the nearest neighbors.
    /// Must be passed the query element `q`.
    fn initialize_searcher(&self, q: &T, searcher: &mut Searcher<Met::Unit>) {
        // Clear the searcher.
        searcher.clear();
        // Add the entry point.
        //
        // NOTE: this seed is deliberately NOT deletion-aware, unlike the public
        // [`Hnsw::entry`]. A tombstone is still a perfectly good *navigational*
        // waypoint — its feature and its edges are intact, and the upper-layer
        // descent never consults `is_deleted` — so seeding from a dead node costs
        // nothing, while skipping to a different top-layer node lands the greedy
        // (`cap == 1`) descent in a worse basin. Measured on a clustered corpus
        // (8 clusters x 50 pts, ef=24, k=10, 40 trials), picking the first LIVE
        // top-layer node instead was never better and up to ~1.1pp worse in
        // recall@10 once the top of the hierarchy was deleted. What did matter is
        // that a dead seed must not occupy a slot in the bounded result heap —
        // see the scrub in `search_layer`.
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

    /// Gets the entry point's feature.
    fn entry_feature(&self) -> &T {
        if let Some(last_layer) = self.layers.last() {
            self.features.get_feature(last_layer[0].zero_node)
        } else {
            self.features.get_feature(0)
        }
    }

    /// Generates a correctly distributed random level as per Algorithm 1 line 4 of the paper.
    fn random_level(&mut self) -> usize {
        let uniform: f64 = self.prng.next_u64() as f64 / u64::MAX as f64;
        (-libm::log(uniform) * libm::log(M as f64).recip()) as usize
    }

    /// Creates a new node at a layer given its nearest neighbors in that layer.
    /// This contains Algorithm 3 from the paper, but also includes some additional logic.
    fn create_node(&mut self, q: &T, nearest: &[Neighbor<Met::Unit>], layer: usize) {
        if layer == 0 {
            let new_index = self.zero.len();
            let mut neighbors: [usize; M0] = [!0; M0];
            for (d, s) in neighbors.iter_mut().zip(nearest.iter()) {
                *d = s.index;
            }
            let node = NeighborNodes { neighbors };
            for neighbor in node.get_neighbors() {
                self.add_neighbor(q, new_index, neighbor, layer);
            }
            self.zero.push(node);
        } else {
            let new_index = self.layers[layer - 1].len();
            let mut neighbors: [usize; M] = [!0; M];
            for (d, s) in neighbors.iter_mut().zip(nearest.iter()) {
                *d = s.index;
            }
            let node = Node {
                zero_node: self.zero.len(),
                next_node: if layer == 1 {
                    self.zero.len()
                } else {
                    self.layers[layer - 2].len()
                },
                neighbors: NeighborNodes { neighbors },
            };
            for neighbor in node.get_neighbors() {
                self.add_neighbor(q, new_index, neighbor, layer);
            }
            self.layers[layer - 1].push(node);
        }
    }

    /// Attempts to add a neighbor to a target node.
    ///
    /// When the target still has a free slot the neighbor simply goes in it.
    /// Once the list is full the surviving set is chosen by the neighbor
    /// selection heuristic (Algorithm 4 of Malkov & Yashunin) rather than by
    /// keeping the nearest `M`.
    ///
    /// Keeping the nearest `M` is correct-looking and pathological on clustered
    /// data: every member of a dense cluster is nearer to every other member
    /// than to anything outside it, so each list saturates with its own cluster
    /// and every outbound link is evicted. The cluster becomes a closed
    /// component that a search can enter but never leave, and points outside it
    /// stop being reachable at any `ef`. The heuristic keeps a candidate when it
    /// is closer to the target than to any neighbor already kept, so one link
    /// survives per direction instead of `M` links into the nearest blob.
    fn add_neighbor(&mut self, q: &T, node_ix: usize, target_ix: usize, layer: usize) {
        let capacity = if layer == 0 { M0 } else { M };

        // Snapshot the current neighbors so the immutable borrows end before the
        // write-back. Filled slots are always a prefix, so the count is also the
        // first free slot.
        let existing: Vec<usize> = if layer == 0 {
            self.zero[target_ix].neighbors[..]
                .iter()
                .copied()
                .take_while(|&n| n != !0)
                .collect()
        } else {
            self.layers[layer - 1][target_ix].neighbors.neighbors[..]
                .iter()
                .copied()
                .take_while(|&n| n != !0)
                .collect()
        };

        if existing.len() < capacity {
            let slot = existing.len();
            if layer == 0 {
                self.zero[target_ix].neighbors[slot] = node_ix;
            } else {
                self.layers[layer - 1][target_ix].neighbors.neighbors[slot] = node_ix;
            }
            return;
        }

        // Borrow every feature involved up front, then work in terms of offsets
        // into that list. The newcomer's feature is not in `self.features` yet —
        // that push happens once the whole node is wired up — so it comes from
        // `q`.
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
                // Keep it only if it is nearer to the target than to anything
                // already kept: that is what makes it a link in a new direction
                // rather than a duplicate of one already held.
                let diverse = kept
                    .iter()
                    .all(|&r| distance_to_target < self.metric.distance(candidate, feature_of(r)));
                if diverse {
                    kept.push(ix);
                } else {
                    pruned.push(ix);
                }
            }

            // Rather than leave slots empty, refill from the pruned candidates
            // in ascending distance — the paper's `keepPrunedConnections`.
            // Degree stays at capacity, so diversity is never bought with
            // connectivity.
            for ix in pruned {
                if kept.len() == capacity {
                    break;
                }
                kept.push(ix);
            }
            kept
        };

        let slots: &mut [usize] = if layer == 0 {
            &mut self.zero[target_ix].neighbors[..]
        } else {
            &mut self.layers[layer - 1][target_ix].neighbors.neighbors[..]
        };
        for (slot, value) in slots
            .iter_mut()
            .zip(kept.into_iter().chain(core::iter::repeat(!0)))
        {
            *slot = value;
        }
    }
}

impl<Met, T, R, const M: usize, const M0: usize> Default for Hnsw<Met, T, R, M, M0>
where
    R: RngCore + SeedableRng,
    Met: Default,
{
    fn default() -> Self {
        Self::new(Met::default())
    }
}
