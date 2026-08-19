//! Pluggable feature storage: an index backed by an external, borrow-stable
//! store must produce exactly the same graph and the same search results as the
//! default `Vec<T>`-backed one.
//!
//! ## What this pins
//!
//! [`hnsw::FeatureStore`] hands out `&T` and requires that a returned reference
//! stays valid — unmoved and unmutated — while later `get` calls happen. That
//! contract is not decorative: the neighbor-selection heuristic (Algorithm 4)
//! holds THREE feature references live at once while a list is being pruned —
//! the target, the candidate under consideration, and each already-kept neighbor
//! it is compared against. A store that returns a reference into a reused
//! scratch buffer therefore aliases all three onto the same memory, every
//! distance collapses toward `distance(x, x) == 0`, and the diversity heuristic
//! silently degenerates back into the nearest-M truncation that closed dense
//! clusters off from the rest of the graph.
//!
//! ## Why these are real tests
//!
//! The store here is a real `mmap` over a real file on disk, driven through the
//! real `insert` / `nearest` paths. The corpora are clustered and large enough
//! that neighbor lists SATURATE, which is the only condition under which the
//! pruning path — the code that depends on the borrow contract — executes at
//! all. Results are compared against the `Vec`-backed index element by element,
//! so any divergence in graph construction surfaces as a different ranked list
//! rather than as a passing test.

//! ## Why this is native-only
//!
//! Every test here is backed by `memmap2` over a real file, and a memory mapping
//! is a capability WASI does not provide — there is no `mmap` to call. The file
//! is gated rather than adapted because the thing under test genuinely does not
//! exist on that target, not because it was inconvenient to run: `store_path`
//! also reaches for `std::env::temp_dir()`, which is an unsupported stub that
//! panics under `wasm32-wasip2`.
//!
//! The borrow-stability contract itself is not wasm-specific and is exercised on
//! every target through the default `Vec<T>` store used by the rest of the
//! suite.

#![cfg(not(target_family = "wasm"))]

use std::fs::OpenOptions;
use std::path::{Path, PathBuf};

use hnsw::{FeatureStore, Hnsw, HnswRuntime, Searcher};
use memmap2::MmapMut;
use rand::{Rng, SeedableRng};
use rand_pcg::Pcg64;
use space::{Metric, Neighbor};

const DIM: usize = 8;
const M: usize = 12;
const M0: usize = 24;

struct Euclidean;

impl Metric<[f32; DIM]> for Euclidean {
    type Unit = u64;
    fn distance(&self, a: &[f32; DIM], b: &[f32; DIM]) -> u64 {
        a.iter()
            .zip(b.iter())
            .map(|(&a, &b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt()
            .to_bits() as u64
    }
}

/// A real file-backed feature store: the features live in an `mmap`, not on the
/// heap, and `get` borrows directly out of the mapping.
///
/// This satisfies the [`FeatureStore`] contract because every index has its own
/// fixed address inside the mapping, so any number of returned references may be
/// live at once and none of them is disturbed by a later `get` or `push`.
struct MmapFeatureStore<const N: usize> {
    mmap: MmapMut,
    len: usize,
    capacity: usize,
    /// Kept so the backing file outlives the mapping for the test's duration.
    _path: PathBuf,
}

impl<const N: usize> MmapFeatureStore<N> {
    const FEATURE_BYTES: usize = N * core::mem::size_of::<f32>();

    fn new(path: impl AsRef<Path>, capacity: usize) -> std::io::Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&path)?;
        file.set_len((capacity * Self::FEATURE_BYTES) as u64)?;
        let mmap = unsafe { MmapMut::map_mut(&file)? };
        assert_eq!(
            mmap.as_ptr() as usize % core::mem::align_of::<[f32; N]>(),
            0,
            "mapping base must be aligned for [f32; N]"
        );
        Ok(Self {
            mmap,
            len: 0,
            capacity,
            _path: path,
        })
    }
}

impl<const N: usize> FeatureStore<[f32; N]> for MmapFeatureStore<N> {
    fn get_feature(&self, index: usize) -> &[f32; N] {
        assert!(
            index < self.len,
            "feature index {index} out of bounds (len {})",
            self.len
        );
        let offset = index * Self::FEATURE_BYTES;
        let bytes = &self.mmap[offset..offset + Self::FEATURE_BYTES];
        // SAFETY: `bytes` is a bounds-checked slice of exactly `FEATURE_BYTES`.
        // The mapping base is page-aligned and `offset` is a multiple of
        // `size_of::<f32>()`, so the pointer is aligned for `[f32; N]`; `f32`
        // has no invalid bit patterns. The mapping is owned by `self` and is
        // never moved or reallocated, so the reference is valid for `&self`.
        unsafe { &*(bytes.as_ptr() as *const [f32; N]) }
    }

    fn push_feature(&mut self, feature: [f32; N]) {
        assert!(
            self.len < self.capacity,
            "MmapFeatureStore capacity {} exceeded",
            self.capacity
        );
        let offset = self.len * Self::FEATURE_BYTES;
        for (i, v) in feature.iter().enumerate() {
            let at = offset + i * core::mem::size_of::<f32>();
            self.mmap[at..at + core::mem::size_of::<f32>()].copy_from_slice(&v.to_ne_bytes());
        }
        self.len += 1;
    }

    fn feature_count(&self) -> usize {
        self.len
    }
}

fn store_path(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("hnsw_feature_store_{name}.bin"))
}

/// Four dense clusters. Each is far larger than `M0`, so every member's zero
/// layer list saturates and the Algorithm 4 pruning path runs for real.
fn clustered_corpus() -> Vec<[f32; DIM]> {
    let mut rng = Pcg64::seed_from_u64(0xC1051E5);
    let centers = [-60.0f32, -20.0, 20.0, 60.0];
    let mut out = Vec::new();
    for &c in &centers {
        for _ in 0..40 {
            let mut v = [0.0f32; DIM];
            for (d, slot) in v.iter_mut().enumerate() {
                *slot = c + (d as f32) * 0.05 + rng.gen_range(-1.0f32..1.0);
            }
            out.push(v);
        }
    }
    out
}

fn probes(features: &[[f32; DIM]]) -> Vec<[f32; DIM]> {
    let mut rng = Pcg64::seed_from_u64(7);
    features
        .iter()
        .copied()
        .chain((0..50).map(|_| {
            let mut v = [0.0f32; DIM];
            for slot in v.iter_mut() {
                *slot = rng.gen_range(-80.0f32..80.0);
            }
            v
        }))
        .collect()
}

fn knn_const<const CM: usize, const CM0: usize, S: FeatureStore<[f32; DIM]>>(
    hnsw: &Hnsw<Euclidean, [f32; DIM], Pcg64, CM, CM0, S>,
    q: &[f32; DIM],
    ef: usize,
    k: usize,
) -> Vec<Neighbor<u64>> {
    let mut searcher = Searcher::default();
    let mut dest = vec![
        Neighbor {
            index: !0,
            distance: !0
        };
        k
    ];
    hnsw.nearest(q, ef, &mut searcher, &mut dest).to_vec()
}

fn knn_runtime<S: FeatureStore<[f32; DIM]>>(
    hnsw: &HnswRuntime<Euclidean, [f32; DIM], Pcg64, S>,
    q: &[f32; DIM],
    ef: usize,
    k: usize,
) -> Vec<Neighbor<u64>> {
    let mut searcher = Searcher::default();
    let mut dest = vec![
        Neighbor {
            index: !0,
            distance: !0
        };
        k
    ];
    hnsw.nearest(q, ef, &mut searcher, &mut dest).to_vec()
}

/// The const-generic index backed by a real mmap must build the same graph and
/// return the same ranked lists as the `Vec`-backed one, on data whose neighbor
/// lists saturate.
#[test]
fn mmap_store_matches_vec_store_on_saturated_clustered_data() {
    let features = clustered_corpus();

    let mut vec_searcher = Searcher::default();
    let mut vec_hnsw: Hnsw<Euclidean, [f32; DIM], Pcg64, M, M0> =
        Hnsw::new_params_and_prng(Euclidean, Default::default(), Pcg64::seed_from_u64(42));
    for f in &features {
        vec_hnsw.insert(*f, &mut vec_searcher);
    }

    let store = MmapFeatureStore::<DIM>::new(store_path("const"), features.len())
        .expect("create mmap store");
    let mut mmap_searcher = Searcher::default();
    let mut mmap_hnsw: Hnsw<Euclidean, [f32; DIM], Pcg64, M, M0, MmapFeatureStore<DIM>> =
        Hnsw::new_with_storage_and_params(
            Euclidean,
            store,
            Default::default(),
            Pcg64::seed_from_u64(42),
        );
    for f in &features {
        mmap_hnsw.insert(*f, &mut mmap_searcher);
    }

    assert_eq!(vec_hnsw.len(), mmap_hnsw.len(), "index sizes must match");

    let ef = 64;
    let k = 20;
    for (qi, q) in probes(&features).iter().enumerate() {
        let a = knn_const(&vec_hnsw, q, ef, k);
        let b = knn_const(&mmap_hnsw, q, ef, k);
        assert_eq!(
            a, b,
            "mmap-backed const index diverged from Vec-backed on probe {qi}"
        );
    }
}

/// Same guarantee for the runtime-degree index.
#[test]
fn runtime_mmap_store_matches_vec_store_on_saturated_clustered_data() {
    let features = clustered_corpus();

    let mut vec_searcher = Searcher::default();
    let mut vec_hnsw: HnswRuntime<Euclidean, [f32; DIM], Pcg64> = HnswRuntime::new_params_and_prng(
        Euclidean,
        M,
        M0,
        Default::default(),
        Pcg64::seed_from_u64(42),
    );
    for f in &features {
        vec_hnsw.insert(*f, &mut vec_searcher);
    }

    let store = MmapFeatureStore::<DIM>::new(store_path("runtime"), features.len())
        .expect("create mmap store");
    let mut mmap_searcher = Searcher::default();
    let mut mmap_hnsw: HnswRuntime<Euclidean, [f32; DIM], Pcg64, MmapFeatureStore<DIM>> =
        HnswRuntime::new_with_storage_and_params(
            Euclidean,
            M,
            M0,
            store,
            Default::default(),
            Pcg64::seed_from_u64(42),
        );
    for f in &features {
        mmap_hnsw.insert(*f, &mut mmap_searcher);
    }

    assert_eq!(vec_hnsw.len(), mmap_hnsw.len(), "index sizes must match");

    let ef = 64;
    let k = 20;
    for (qi, q) in probes(&features).iter().enumerate() {
        let a = knn_runtime(&vec_hnsw, q, ef, k);
        let b = knn_runtime(&mmap_hnsw, q, ef, k);
        assert_eq!(
            a, b,
            "mmap-backed runtime index diverged from Vec-backed on probe {qi}"
        );
    }
}

/// Cross-cluster reachability must survive external storage. This is the
/// clustered-recall guarantee driven through a store other than `Vec`: three
/// points beside the origin and a dense far cluster that saturates every list.
/// If the storage port broke the borrow contract the pruning heuristic would
/// degenerate and the near points would become unreachable.
#[test]
fn mmap_store_preserves_cross_cluster_reachability() {
    let mut features: Vec<[f32; DIM]> = Vec::new();
    for c in [1.0f32, 2.0, 3.0] {
        let mut v = [0.0f32; DIM];
        v[0] = c;
        features.push(v);
    }
    for i in 0..37 {
        let mut v = [0.0f32; DIM];
        v[0] = 500.0 + i as f32;
        v[1] = 500.0;
        features.push(v);
    }

    let store = MmapFeatureStore::<DIM>::new(store_path("reach"), features.len())
        .expect("create mmap store");
    let mut searcher = Searcher::default();
    let mut hnsw: Hnsw<Euclidean, [f32; DIM], Pcg64, 16, 32, MmapFeatureStore<DIM>> =
        Hnsw::new_with_storage(Euclidean, store, Pcg64::seed_from_u64(42));
    for f in &features {
        hnsw.insert(*f, &mut searcher);
    }

    let origin = [0.0f32; DIM];
    let got = knn_const::<16, 32, MmapFeatureStore<DIM>>(&hnsw, &origin, 200, 3);

    assert_eq!(got.len(), 3, "must return three neighbors");
    for n in &got {
        let d = f32::from_bits(n.distance as u32);
        assert!(
            d < 10.0,
            "a dense far cluster hid the near points through mmap storage: got \
             distance {}, expected the points at distance 1, 2 and 3",
            d
        );
    }
}

/// The borrow-stability contract itself: references handed out by a conforming
/// store remain valid and distinct while further `get` calls are made.
///
/// This is exactly the pattern the pruning heuristic relies on. A store that
/// returned a reference into a single reused scratch buffer would alias these
/// handles onto the same address, and the final assertion — that a real distance
/// between two different features is non-zero — would collapse to zero.
#[test]
fn concurrently_live_feature_handles_stay_valid() {
    let features = clustered_corpus();
    let mut store = MmapFeatureStore::<DIM>::new(store_path("handles"), features.len())
        .expect("create mmap store");
    for f in &features {
        store.push_feature(*f);
    }

    // One from the first cluster, one from the last: far apart by construction.
    let a_ix = 0usize;
    let b_ix = features.len() - 1;

    let a = store.get_feature(a_ix);
    let b = store.get_feature(b_ix);

    // Both handles are live simultaneously; a further read must not disturb them.
    let c = store.get_feature(features.len() / 2);

    assert_ne!(
        a.as_ptr(),
        b.as_ptr(),
        "each index must have its own stable address"
    );
    assert_ne!(a.as_ptr(), c.as_ptr(), "third read aliased the first");

    assert_eq!(
        a, &features[a_ix],
        "first handle was disturbed by later reads"
    );
    assert_eq!(
        b, &features[b_ix],
        "second handle was disturbed by later reads"
    );

    let metric = Euclidean;
    let d = metric.distance(a, b);
    assert_eq!(
        d,
        metric.distance(&features[a_ix], &features[b_ix]),
        "distance over two live store handles must equal the direct distance"
    );
    assert_ne!(
        f32::from_bits(d as u32),
        0.0,
        "two distinct features must not measure as identical — the handles aliased"
    );
}
