//! Pluggable storage for the feature vectors an index is built over.

use alloc::vec::Vec;

/// Backing storage for an index's feature vectors.
///
/// The default is [`Vec<T>`], which keeps every feature on the heap. Supplying a
/// different implementation lets the features live somewhere else — an `mmap`ed
/// file, an arena, or any other region the caller manages — while the graph
/// itself stays in memory.
///
/// # Contract
///
/// A reference returned by [`get_feature`](FeatureStore::get_feature) must
/// remain valid, unmoved, and unmutated for as long as it is held, regardless of
/// any later calls to `get_feature`. Equivalently: each index must denote its own
/// storage location with a stable address, and reads must not disturb one
/// another.
///
/// This is load-bearing rather than a formality. The neighbor-selection
/// heuristic keeps up to three feature references live simultaneously while
/// pruning a saturated neighbor list — the target node, the candidate being
/// considered, and each already-kept neighbor the candidate is compared
/// against. An implementation that decoded features into a shared scratch
/// buffer would alias all three onto one address; every comparison would
/// degenerate toward `distance(x, x)`, the diversity heuristic would silently
/// collapse into nearest-M truncation, and dense clusters would close
/// themselves off from the rest of the graph.
///
/// Suitable backings therefore include `Vec<T>`, a fixed `mmap` region, an
/// arena, or an append-only cache that never evicts. Backings that reuse a
/// decode buffer, evict entries, or synthesize values per read cannot uphold
/// this contract and are not supported.
///
/// # Naming
///
/// The methods are deliberately named `get_feature` / `push_feature` /
/// `feature_count` rather than `get` / `push` / `len`. A trait method named
/// `get` on `Vec<T>` shadows `slice::get` at every call site where this trait is
/// in scope, because trait methods on the receiver type are considered before
/// inherent methods reached through `Deref`. The verbose names keep `Vec`'s own
/// API usable in the same module.
///
/// # Panics
///
/// Implementations are expected to panic on an out-of-bounds index, matching
/// `Vec`'s behavior.
pub trait FeatureStore<T> {
    /// Returns the feature stored at `index`.
    fn get_feature(&self, index: usize) -> &T;

    /// Appends a feature, giving it the next index.
    fn push_feature(&mut self, feature: T);

    /// The number of features stored.
    fn feature_count(&self) -> usize;
}

impl<T> FeatureStore<T> for Vec<T> {
    #[inline]
    fn get_feature(&self, index: usize) -> &T {
        &self[index]
    }

    #[inline]
    fn push_feature(&mut self, feature: T) {
        Vec::push(self, feature)
    }

    #[inline]
    fn feature_count(&self) -> usize {
        Vec::len(self)
    }
}
