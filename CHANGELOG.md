# Changelog

All notable changes to this fork are recorded here.

This is a fork of [`rust-cv/hnsw`](https://github.com/rust-cv/hnsw). Everything
below is relative to upstream `0.11`, and every entry links the pull request
that carries the reasoning and the measurements behind it.

Read the **Behavioral impact** notes before upgrading. Several entries change
search results, and one changes the graph that is built, which matters if any
index was serialized by an earlier version.

## [0.12.0-alpha.0] — unreleased

### Fixed

- **The serialized format depended on the writer's pointer width.** An unused
  neighbour slot is `!0usize` in memory, so it was written as
  `0xFFFF_FFFF_FFFF_FFFF` by a 64-bit build and `0xFFFF_FFFF` by a 32-bit one.
  Both directions were broken: a 64-bit-written index FAILED to load on
  `wasm32` (`invalid value: integer 18446744073709551615, expected usize`),
  and a 32-bit-written index loaded on 64-bit with each empty slot silently
  reinterpreted as neighbour index `4294967295`, because the `take_while(|&n| n
  != !0)` terminator no longer matched. The wire sentinel is now pinned at
  `u64::MAX` and mapped back to this target's `!0` on read, which leaves 64-bit
  output byte-identical so existing indexes stay readable.
  ([#11](https://github.com/npiesco/hnsw/pull/11))

- **A dense cluster could close itself off entirely.** When a neighbor list was
  full, `add_neighbor` kept the nearest `M`. On clustered data every member of a
  cluster is nearer to every other member than to anything outside it, so each
  list saturated with the cluster's own members and every outbound link was
  evicted. The failure was total rather than marginal: points could not find
  themselves. Replaced with the diversity heuristic of Malkov & Yashunin
  Algorithm 4, plus `keepPrunedConnections` refill so degree is unchanged.
  ([#1](https://github.com/npiesco/hnsw/pull/1))

- **A tombstoned navigation seed consumed a result slot.** `initialize_searcher`
  and `lower_search` push the entry point and the carried candidate into
  `nearest` without a liveness check, and the only scrub ran after the zero-layer
  search. For the whole of that search a deleted node occupied one of the `ef`
  slots, so results under-filled by exactly one. The scrub moved ahead of the
  search; `candidates` deliberately still carries tombstones, so live nodes
  reachable only through a deleted node stay reachable.
  ([#3](https://github.com/npiesco/hnsw/pull/3))

- **`layer_item_id` panicked for every non-zero level.** It indexed the wrong
  layer — a 100% failure rate on that path, also present upstream.
  ([#4](https://github.com/npiesco/hnsw/pull/4))

- **Soft-deleted nodes bypassed the search work budget.** Deleted nodes were
  pushed to `candidates` unconditionally while live nodes were gated on the beam
  being unsaturated, so a search with tombstones present visited ~101% of the
  corpus at every deletion density. It hid behind perfect recall: recall was
  1.0 everywhere precisely because the search was scanning everything and
  ignoring `ef`. ([#7](https://github.com/npiesco/hnsw/pull/7))

### Added

- **Filtered search: `nearest_filtered`.** Applies a caller predicate DURING
  traversal rather than filtering results afterwards, so a selective predicate
  still fills `k`. Work scales as roughly `ef / selectivity`, which is inherent
  to filtered ANN — see the budget notes on the method.
  ([#5](https://github.com/npiesco/hnsw/pull/5))

- **Configurable `level_scale`.** Controls the level-assignment distribution, so
  the hierarchy depth can be tuned. Layer allocation is capped, so an extreme
  value cannot allocate unbounded layers. Params keep the legacy bincode layout
  and payloads without the field still deserialize.
  ([#5](https://github.com/npiesco/hnsw/pull/5),
  [#6](https://github.com/npiesco/hnsw/pull/6))

- **Pluggable feature storage: the `FeatureStore<T>` trait.** Lets features live
  outside the heap. Borrow-based — `get_feature` returns `&T` — because the
  diversity heuristic holds three feature references live at once while pruning
  a saturated list, so a shared-scratch-buffer design silently turns
  `distance(a, b)` into `distance(x, x)`. `Hnsw` and `HnswRuntime` take a
  trailing storage parameter defaulting to `Vec<T>`, so existing signatures are
  unchanged. ([#2](https://github.com/npiesco/hnsw/pull/2))

- **Soft delete on the runtime-degree index.** `HnswRuntime` gained the
  `mark_delete` support the const-generic index already had.
  ([#2](https://github.com/npiesco/hnsw/pull/2))

### Changed

- **Zero-layer QUERIES now traverse best-first.** They previously drained
  `Searcher.candidates`, a LIFO `Vec`, giving a depth-first traversal with no
  termination check — a deviation from Algorithm 2 of the paper. Measured
  against distance evaluations rather than at equal `ef`, best-first reached
  higher recall for less work at every comparable point: at N=20,000 it reached
  0.7700 recall in 2,573 evaluations where depth-first needed 4,323 to reach
  0.7800 and managed only 0.6940 by 3,154.

  The comparison at equal `ef` looks the other way round, because `ef` bounds the
  RESULT LIST rather than the expansion, and the depth-first traversal simply
  does much more work for the same `ef`. The recall-per-evaluation curve is the
  honest comparison.

  The INSERT path deliberately remains depth-first; changing it would change
  every graph built. ([#9](https://github.com/npiesco/hnsw/pull/9))

- **Migrated to edition 2024, and the crate now builds under `-D warnings`.**
  ([#5](https://github.com/npiesco/hnsw/pull/5))

### Behavioral impact

Stated conditionally, because these do not all apply to every workload:

- **Graph construction can change** with
  [#1](https://github.com/npiesco/hnsw/pull/1). The new neighbour selection only
  differs once a neighbour list SATURATES, so a corpus that never fills one
  builds an identical graph. Where pruning is exercised, an index built after the
  change differs from one built before it from the same input. Previously
  serialized indexes remain loadable — the format did not change — but they no
  longer match a fresh rebuild.

- **Search results change for newly built indexes** with
  [#1](https://github.com/npiesco/hnsw/pull/1), because the graph differs. An
  index serialized BEFORE that change and queried by new code is unaffected by
  it: the links are already baked in.

- **Search results change for every query** with
  [#9](https://github.com/npiesco/hnsw/pull/9), which alters the query traversal
  itself and so applies to old and new graphs alike. Anything pinning golden
  query outputs must re-derive them.

- **Tombstone workloads only** for [#3](https://github.com/npiesco/hnsw/pull/3)
  and [#7](https://github.com/npiesco/hnsw/pull/7). An index that has never had
  `mark_delete` called on it sees no change from either.

- **The same `ef` now performs less expansion** after
  [#9](https://github.com/npiesco/hnsw/pull/9), and may therefore give lower
  recall at equal `ef`. `ef` bounds the result list, not the work, and the
  previous depth-first traversal over-expanded — which is why it scored better at
  equal `ef` and worse per distance evaluation. A consumer that tuned `ef`
  against the old traversal should re-derive it against measured recall rather
  than carrying the old number over.

- **Indexes are portable across pointer widths** as of
  [#11](https://github.com/npiesco/hnsw/pull/11). 64-bit output is unchanged, so
  nothing needs re-writing. An index written by a 32-bit build BEFORE that change
  is the one exception: its empty neighbour slots were recorded as `4294967295`,
  which is indistinguishable from a real index of that value, so such a file
  cannot be repaired and must be rebuilt.

### Testing notes

- [#8](https://github.com/npiesco/hnsw/pull/8) pinned the rare-filter regime and
  corrected a false recall contract in this repository's own test suite — an
  assertion that could not fail.
- [#6](https://github.com/npiesco/hnsw/pull/6) added `level_scale` coverage for
  the runtime-degree index, which had none.
- [#10](https://github.com/npiesco/hnsw/pull/10) corrected traversal claims that
  went stale with [#9](https://github.com/npiesco/hnsw/pull/9), fixed ten broken
  rustdoc links, narrowed `runtime_parity.rs`'s documented scope to what it
  actually checks, and strengthened the accept-all filtered test from
  `recall > 0.5` to exact equality with unfiltered search — a contract that only
  became available once both paths shared one traversal.

### Known gaps

- **Nothing pins the graph itself.** `tests/runtime_parity.rs` compares the
  const-generic and runtime implementations against each other, so a change
  applied to both — which any change to shared insert behaviour must be — leaves
  it green. There is no golden serialized fixture and no graph fingerprint, so
  "graph construction changed" above is reasoned from the code rather than
  caught by a test.
