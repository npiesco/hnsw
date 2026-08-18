use crate::hnsw::nodes::NeighborNodes;
use core::fmt;
use serde::{
    Deserialize, Deserializer, Serialize, Serializer,
    de::{Error, Expected, SeqAccess, Visitor},
    ser::SerializeSeq,
};

/// Wire value for an unused neighbour slot.
///
/// In memory that slot is `!0usize`, whose width follows the target: it is
/// `0xFFFF_FFFF_FFFF_FFFF` on a 64-bit build and `0xFFFF_FFFF` on a 32-bit one
/// such as `wasm32`. Writing the in-memory value directly therefore produces a
/// file whose meaning depends on the machine that wrote it, in BOTH directions:
///
/// * **64-bit writer, 32-bit reader** — deserialization FAILS outright with
///   `invalid value: integer 18446744073709551615, expected usize`, because the
///   value does not fit a 32-bit `usize`. An index built on a server cannot be
///   opened in a browser at all.
/// * **32-bit writer, 64-bit reader** — deserialization SUCCEEDS and is wrong.
///   `4294967295` is perfectly representable as a 64-bit `usize`, so the
///   terminator in `get_neighbors`' `take_while(|&n| n != !0)` never matches and
///   the slot is treated as a real neighbour index. That is silent graph
///   corruption rather than an error.
///
/// Pinning the wire sentinel at `u64::MAX` fixes both. It is also exactly what a
/// 64-bit build already wrote, so every index serialized by a 64-bit build up to
/// now stays byte-identical and readable.
const WIRE_EMPTY: u64 = u64::MAX;

impl<const N: usize> Serialize for NeighborNodes<N> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        // Elements are written as `u64` rather than `usize` so the encoding does
        // not depend on the pointer width of the writing machine. On a 64-bit
        // build this is byte-for-byte what `self.neighbors[..].serialize(..)`
        // produced before, since `usize` already encodes as `u64` there.
        let mut seq = serializer.serialize_seq(Some(N))?;
        for &n in &self.neighbors {
            let wire = if n == !0 { WIRE_EMPTY } else { n as u64 };
            seq.serialize_element(&wire)?;
        }
        seq.end()
    }
}

impl<'de, const N: usize> Deserialize<'de> for NeighborNodes<N> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_seq(NeighborNodesVisitor::<N>)
    }
}

struct NeighborNodesVisitor<const N: usize>;

impl<'de, const N: usize> Visitor<'de> for NeighborNodesVisitor<N> {
    type Value = NeighborNodes<N>;

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(formatter, "[usize; {}]", N)
    }

    fn visit_seq<S>(self, mut seq: S) -> Result<NeighborNodes<N>, S::Error>
    where
        S: SeqAccess<'de>,
    {
        let mut neighbors = [!0; N];
        let mut position = 0;

        // Read as `u64` for the reason given on `WIRE_EMPTY`. Reading as `usize`
        // is what made a 32-bit reader reject the sentinel a 64-bit writer had
        // produced, which is the whole defect.
        while let Some(n) = seq.next_element::<u64>()? {
            if position < N {
                neighbors[position] = if n == WIRE_EMPTY {
                    !0
                } else {
                    usize::try_from(n).map_err(|_| {
                        Error::custom(format_args!(
                            "neighbour index {n} does not fit this target's \
                             usize; the index was built on a machine with a \
                             wider pointer than this one"
                        ))
                    })?
                };
                position += 1;
            } else {
                return Err(Error::invalid_length(
                    N + 1,
                    &NeighborNodesExpectedNum::<N>(true),
                ));
            }
        }

        if position != N {
            Err(Error::invalid_length(
                position,
                &NeighborNodesExpectedNum::<N>(false),
            ))
        } else {
            Ok(NeighborNodes { neighbors })
        }
    }
}

struct NeighborNodesExpectedNum<const N: usize>(bool);

impl<const N: usize> Expected for NeighborNodesExpectedNum<N> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(
            formatter,
            "{} elements was expected; found {}",
            N,
            if self.0 { "too many" } else { "too few" }
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;
    use bincode::Options;

    fn options() -> impl Options {
        bincode::DefaultOptions::new()
            .with_little_endian()
            .with_fixint_encoding()
    }

    /// The encoding is pinned to an EXACT byte sequence rather than checked by
    /// round trip, because a round trip cannot see this defect: it is symmetric
    /// on any single target and only breaks when the writer and reader have
    /// different pointer widths.
    ///
    /// On a 64-bit build this is what was already written, so the format is
    /// unchanged and existing indexes stay readable. On `wasm32` the previous
    /// code emitted four-byte elements and a four-byte sentinel, producing 24
    /// bytes here instead of 40 — an incompatible file.
    #[test]
    fn a_neighbour_list_encodes_as_length_prefixed_u64s() {
        let nodes = NeighborNodes::<4> {
            neighbors: [!0, 7, !0, !0],
        };
        let bytes = options().serialize(&nodes).expect("serialize");

        let mut expected = Vec::new();
        expected.extend_from_slice(&4u64.to_le_bytes()); // sequence length
        expected.extend_from_slice(&u64::MAX.to_le_bytes()); // empty slot
        expected.extend_from_slice(&7u64.to_le_bytes()); // real neighbour
        expected.extend_from_slice(&u64::MAX.to_le_bytes());
        expected.extend_from_slice(&u64::MAX.to_le_bytes());

        assert_eq!(
            bytes, expected,
            "neighbour lists must encode as a u64 length followed by u64 \
             elements, with `u64::MAX` for an empty slot, on every target"
        );
    }

    /// The sentinel must come back as this target's `!0usize`, not as the
    /// literal number that was on the wire.
    #[test]
    fn the_wire_sentinel_decodes_to_this_targets_empty_marker() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&4u64.to_le_bytes());
        bytes.extend_from_slice(&u64::MAX.to_le_bytes());
        bytes.extend_from_slice(&3u64.to_le_bytes());
        bytes.extend_from_slice(&u64::MAX.to_le_bytes());
        bytes.extend_from_slice(&u64::MAX.to_le_bytes());

        let nodes: NeighborNodes<4> = options().deserialize(&bytes).expect("deserialize");

        assert_eq!(nodes.neighbors[0], !0, "sentinel must decode to !0");
        assert_eq!(nodes.neighbors[1], 3, "real index must decode unchanged");
        assert_eq!(nodes.neighbors[2], !0);
        assert_eq!(nodes.neighbors[3], !0);
    }

    /// A real index too large for this target must fail loudly rather than wrap
    /// or truncate into a plausible-looking neighbour.
    ///
    /// Only observable on a 32-bit target; on 64-bit the value fits and decodes
    /// normally, which is correct behaviour there.
    #[test]
    fn an_out_of_range_neighbour_index_is_reported() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&4u64.to_le_bytes());
        // One below the sentinel: a "real" index no 32-bit target can hold.
        bytes.extend_from_slice(&(u64::MAX - 1).to_le_bytes());
        bytes.extend_from_slice(&u64::MAX.to_le_bytes());
        bytes.extend_from_slice(&u64::MAX.to_le_bytes());
        bytes.extend_from_slice(&u64::MAX.to_le_bytes());

        let decoded: Result<NeighborNodes<4>, _> = options().deserialize(&bytes);

        if usize::BITS >= 64 {
            let nodes = decoded.expect("a 64-bit target can hold this index");
            assert_eq!(nodes.neighbors[0], (u64::MAX - 1) as usize);
        } else {
            assert!(
                decoded.is_err(),
                "a neighbour index that does not fit this target's usize must be \
                 an error, not a silently truncated index"
            );
        }
    }
}
