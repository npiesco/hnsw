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

- **Graph construction changed** in [#1](https://github.com/npiesco/hnsw/pull/1)
  (neighbor selection). An index serialized before that change does not match one
  built after it from the same input. Note that `tests/runtime_parity.rs` cannot
  detect this: it compares the two implementations against each other with no
  golden reference, so a change applied to both leaves it green.

- **Search results changed** in [#1](https://github.com/npiesco/hnsw/pull/1),
  [#3](https://github.com/npiesco/hnsw/pull/3),
  [#7](https://github.com/npiesco/hnsw/pull/7) and
  [#9](https://github.com/npiesco/hnsw/pull/9). Consumers that pin golden query
  outputs will need to re-derive them.

- **`ef` buys less work than it used to** after
  [#9](https://github.com/npiesco/hnsw/pull/9), because the traversal no longer
  over-expands. A consumer that tuned `ef` against the old depth-first traversal
  should re-derive it against recall rather than carrying the old number over.

### Testing notes

- [#8](https://github.com/npiesco/hnsw/pull/8) pinned the rare-filter regime and
  corrected a false recall contract in this repository's own test suite — an
  assertion that could not fail.
- [#6](https://github.com/npiesco/hnsw/pull/6) added `level_scale` coverage for
  the runtime-degree index, which had none.
- [#10](https://github.com/npiesco/hnsw/pull/10) corrected traversal claims that
  went stale with [#9](https://github.com/npiesco/hnsw/pull/9), fixed ten broken
  rustdoc links, and narrowed `runtime_parity.rs`'s documented scope to what it
  actually checks.
