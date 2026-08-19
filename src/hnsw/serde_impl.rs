use crate::hnsw::nodes::NeighborNodes;
use core::fmt;
use serde::{
    Deserialize, Deserializer, Serialize, Serializer,
    de::{Error, Expected, SeqAccess, Visitor},
    ser::SerializeSeq,
};

/// Wire value for an unused neighbour slot.
///
/// In memory that slot is `!0usize`, whose VALUE follows the target: it is
/// 18446744073709551615 on a 64-bit build and 4294967295 on a 32-bit one such as
/// `wasm32`. Writing it directly therefore produced a file whose meaning
/// depended on the machine that wrote it, in BOTH directions:
///
/// * **64-bit writer, 32-bit reader** — deserialization FAILS outright with
///   `invalid value: integer 18446744073709551615, expected usize`, because the
///   value does not fit a 32-bit `usize`. An index built on a server could not
///   be opened in a browser at all.
/// * **32-bit writer, 64-bit reader** — deserialization SUCCEEDS and is wrong.
///   4294967295 is perfectly representable as a 64-bit `usize`, so the
///   terminator in `get_neighbors`' `take_while(|&n| n != !0)` never matches and
///   the slot is treated as a real neighbour index. That is silent graph
///   corruption rather than an error.
///
/// Note this is about the sentinel's VALUE, not the encoded width. `serde`
/// routes `usize` through `serialize_u64`, so bincode wrote eight bytes on every
/// target and always has; a 32-bit build emitted `FF FF FF FF 00 00 00 00`, not
/// four bytes. That is asserted by
/// `tests::a_bare_usize_encodes_as_eight_bytes_on_every_target` rather than
/// merely claimed here, because the first version of this comment asserted the
/// width story and was wrong.
///
/// Pinning the wire sentinel at `u64::MAX` fixes both directions, and is exactly
/// what a 64-bit build already wrote, so indexes serialized by a 64-bit build
/// stay byte-identical and readable.
const WIRE_EMPTY: u64 = u64::MAX;

/// Value a 32-bit build wrote for an empty slot before `WIRE_EMPTY` existed.
///
/// Such a file is only produced by a 32-bit writer, and on that target no graph
/// can have 2^32 - 1 nodes, so this value can never be a legitimate neighbour
/// index in it. Treating it as empty therefore recovers those files rather than
/// silently importing a nonexistent neighbour — which is what a reader that
/// simply accepted the number would do, and is the more damaging of the two
/// original failure directions because it does not announce itself.
const LEGACY_32_BIT_EMPTY: u64 = u32::MAX as u64;

impl<const N: usize> Serialize for NeighborNodes<N> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        // Elements are written as `u64` for clarity, but the load-bearing part
        // is mapping the in-memory sentinel onto a FIXED wire value. `serde`
        // already routes `usize` through `serialize_u64`, so the encoded WIDTH
        // was never the problem — see
        // `tests::a_bare_usize_encodes_as_eight_bytes_on_every_target`.
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

        // Read as `u64` and canonicalize the sentinel. Reading as `usize` is
        // what made a 32-bit reader reject the value a 64-bit writer produced.
        while let Some(n) = seq.next_element::<u64>()? {
            if position < N {
                neighbors[position] = if n == WIRE_EMPTY || n == LEGACY_32_BIT_EMPTY {
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

    /// Settles what the encoding width of a bare `usize` actually is.
    ///
    /// This exists because I originally described the defect as `usize` being
    /// written four bytes wide on a 32-bit target. That was WRONG. `serde`'s
    /// `impl Serialize for usize` forwards to `serialize_u64`, so bincode writes
    /// eight bytes on every target and always has. The old encoding of an empty
    /// slot on 32-bit was `FF FF FF FF 00 00 00 00` — eight bytes carrying the
    /// value 4294967295 — not four bytes.
    ///
    /// The defect is therefore VALUE canonicalization, not integer width: the
    /// sentinel's numeric value tracked the writer's pointer width even though
    /// its encoded width did not. Writing `u64` explicitly is clarity; mapping
    /// `!0usize` to `u64::MAX` is the actual fix.
    ///
    /// Asserted rather than asserted-in-prose so the claim is checked on
    /// whichever target the suite runs on.
    #[test]
    fn a_bare_usize_encodes_as_eight_bytes_on_every_target() {
        let one: usize = 1;
        let bytes = options().serialize(&one).expect("serialize");
        assert_eq!(
            bytes.len(),
            8,
            "bincode fixint must encode `usize` as eight bytes; got {bytes:?} on \
             a target with usize::BITS = {}",
            usize::BITS
        );
        assert_eq!(bytes, 1u64.to_le_bytes());
    }

    /// The encoding is pinned to an EXACT byte sequence rather than checked by
    /// round trip, because a round trip cannot see this defect: it is symmetric
    /// on any single target and only breaks when the writer and reader disagree
    /// about what an empty slot's VALUE is.
    ///
    /// Before the fix this produced the same 40 bytes on 64-bit — the format is
    /// unchanged there — but on a 32-bit target the first element was
    /// `FF FF FF FF 00 00 00 00` (4294967295) rather than `FF * 8`.
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

    /// A file written by a 32-bit build BEFORE the sentinel was canonicalized
    /// must be recovered rather than silently importing a phantom neighbour.
    ///
    /// This is the direction that fails quietly: 4294967295 is a representable
    /// 64-bit `usize`, so a reader that simply accepted it would hand back a
    /// neighbour index no node has, and `take_while(|&n| n != !0)` would not
    /// stop there.
    #[test]
    fn a_legacy_32_bit_empty_slot_is_recovered_as_empty() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&4u64.to_le_bytes());
        bytes.extend_from_slice(&2u64.to_le_bytes());
        bytes.extend_from_slice(&(u32::MAX as u64).to_le_bytes());
        bytes.extend_from_slice(&(u32::MAX as u64).to_le_bytes());
        bytes.extend_from_slice(&(u32::MAX as u64).to_le_bytes());

        let nodes: NeighborNodes<4> = options().deserialize(&bytes).expect("deserialize");

        assert_eq!(nodes.neighbors[0], 2, "a real index must survive");
        assert_eq!(
            nodes.neighbors[1], !0,
            "a legacy 32-bit sentinel must be recovered as empty, not imported \
             as neighbour index 4294967295"
        );
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
