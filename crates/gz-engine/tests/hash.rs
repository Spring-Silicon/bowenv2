use gz_engine::{
    ActionSetHash, CandidateHash, EngineId, EngineVersion, GraphHash, HexParseError,
    MeasureConfigHash, ModelVersion, SearchConfigHash,
};
use std::collections::{BTreeSet, HashSet};

macro_rules! assert_id_type {
    ($name:ident, $len:literal) => {{
        let bytes = [0xab; $len];
        let id = $name::from_bytes(bytes);

        assert_eq!(id.as_bytes(), &bytes);
        assert_eq!($name::BYTE_LEN, $len);
        assert_eq!($name::HEX_LEN, $len * 2);

        let hex = "ab".repeat($len);
        assert_eq!(id.to_string(), hex);
        assert_eq!(format!("{id:?}"), format!("{}({hex})", stringify!($name)));
        assert_eq!($name::try_from_hex(&hex).unwrap(), id);
        assert_eq!(hex.parse::<$name>().unwrap(), id);

        let upper_hex = "AB".repeat($len);
        assert_eq!($name::try_from_hex(&upper_hex).unwrap(), id);

        assert_eq!(
            $name::try_from_hex("ab").unwrap_err(),
            HexParseError::InvalidLength {
                expected: $len * 2,
                actual: 2,
            }
        );

        let invalid = format!("{}zz", "ab".repeat($len - 1));
        assert_eq!(
            $name::try_from_hex(&invalid).unwrap_err(),
            HexParseError::InvalidCharacter {
                index: ($len - 1) * 2,
                byte: b'z',
            }
        );
    }};
}

#[test]
fn hash_types_roundtrip_and_reject_invalid_hex() {
    assert_id_type!(GraphHash, 32);
    assert_id_type!(CandidateHash, 32);
    assert_id_type!(ActionSetHash, 32);
    assert_id_type!(MeasureConfigHash, 32);
    assert_id_type!(SearchConfigHash, 32);
}

#[test]
fn version_types_roundtrip_and_reject_invalid_hex() {
    assert_id_type!(EngineId, 16);
    assert_id_type!(EngineVersion, 16);
    assert_id_type!(ModelVersion, 16);
}

#[test]
fn generated_ids_work_as_ordered_and_hashed_keys() {
    let id = GraphHash::from_bytes([1; 32]);

    let mut btree = BTreeSet::new();
    btree.insert(id);
    assert!(btree.contains(&id));

    let mut hash_set = HashSet::new();
    hash_set.insert(id);
    assert!(hash_set.contains(&id));
}

#[cfg(feature = "serde")]
mod serde_tests {
    use super::*;
    use serde_test::{Configure, Token, assert_tokens};

    #[test]
    fn serde_roundtrip_for_hash_type() {
        const BYTES: [u8; 32] = [0xab; 32];
        let id = GraphHash::from_bytes(BYTES);

        assert_tokens(
            &id.readable(),
            &[Token::Str(
                "abababababababababababababababababababababababababababababababab",
            )],
        );
        assert_tokens(&id.compact(), &[Token::Bytes(&BYTES)]);
    }

    #[test]
    fn serde_roundtrip_for_version_type() {
        const BYTES: [u8; 16] = [0xcd; 16];
        let id = EngineVersion::from_bytes(BYTES);

        assert_tokens(
            &id.readable(),
            &[Token::Str("cdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcd")],
        );
        assert_tokens(&id.compact(), &[Token::Bytes(&BYTES)]);
    }
}
