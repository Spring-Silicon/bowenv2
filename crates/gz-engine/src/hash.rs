//! Hash and version identifier types.

use std::fmt;
use std::str::FromStr;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum HexParseError {
    InvalidLength { expected: usize, actual: usize },
    InvalidCharacter { index: usize, byte: u8 },
}

impl fmt::Display for HexParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidLength { expected, actual } => {
                write!(f, "invalid hex length: expected {expected}, got {actual}")
            }
            Self::InvalidCharacter { index, byte } => {
                write!(f, "invalid hex character at byte {index}: 0x{byte:02x}")
            }
        }
    }
}

impl std::error::Error for HexParseError {}

macro_rules! define_id_type {
    ($name:ident, $len:literal) => {
        #[derive(Clone, Copy, Eq, PartialEq, Ord, PartialOrd, Hash)]
        pub struct $name([u8; $len]);

        impl $name {
            pub const BYTE_LEN: usize = $len;
            pub const HEX_LEN: usize = $len * 2;

            #[must_use]
            pub const fn from_bytes(bytes: [u8; $len]) -> Self {
                Self(bytes)
            }

            #[must_use]
            pub const fn as_bytes(&self) -> &[u8; $len] {
                &self.0
            }

            pub fn try_from_hex(hex: &str) -> Result<Self, HexParseError> {
                parse_hex_array::<$len>(hex).map(Self)
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write_hex(f, &self.0)
            }
        }

        impl fmt::Debug for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}(", stringify!($name))?;
                write_hex(f, &self.0)?;
                write!(f, ")")
            }
        }

        impl FromStr for $name {
            type Err = HexParseError;

            fn from_str(s: &str) -> Result<Self, Self::Err> {
                Self::try_from_hex(s)
            }
        }

        #[cfg(feature = "serde")]
        impl serde::Serialize for $name {
            fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
            where
                S: serde::Serializer,
            {
                if serializer.is_human_readable() {
                    serializer.serialize_str(&self.to_string())
                } else {
                    serializer.serialize_bytes(&self.0)
                }
            }
        }

        #[cfg(feature = "serde")]
        impl<'de> serde::Deserialize<'de> for $name {
            fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                let bytes = if deserializer.is_human_readable() {
                    deserializer.deserialize_str(HexVisitor::<$len> {
                        type_name: stringify!($name),
                    })?
                } else {
                    deserializer.deserialize_bytes(HexVisitor::<$len> {
                        type_name: stringify!($name),
                    })?
                };

                Ok(Self(bytes))
            }
        }
    };
}

define_id_type!(GraphHash, 32);
define_id_type!(CandidateHash, 32);
define_id_type!(ActionSetHash, 32);
define_id_type!(MeasureConfigHash, 32);
define_id_type!(SearchConfigHash, 32);
define_id_type!(EngineId, 16);
define_id_type!(EngineVersion, 16);
define_id_type!(ModelVersion, 16);

fn write_hex(f: &mut fmt::Formatter<'_>, bytes: &[u8]) -> fmt::Result {
    for byte in bytes {
        write!(f, "{byte:02x}")?;
    }

    Ok(())
}

fn parse_hex_array<const N: usize>(hex: &str) -> Result<[u8; N], HexParseError> {
    let expected = N * 2;
    let actual = hex.len();

    if actual != expected {
        return Err(HexParseError::InvalidLength { expected, actual });
    }

    let mut bytes = [0u8; N];
    let hex = hex.as_bytes();

    for (index, byte) in bytes.iter_mut().enumerate() {
        let hi_index = index * 2;
        let lo_index = hi_index + 1;
        let hi = hex_value(hex[hi_index]).ok_or(HexParseError::InvalidCharacter {
            index: hi_index,
            byte: hex[hi_index],
        })?;
        let lo = hex_value(hex[lo_index]).ok_or(HexParseError::InvalidCharacter {
            index: lo_index,
            byte: hex[lo_index],
        })?;

        *byte = (hi << 4) | lo;
    }

    Ok(bytes)
}

const fn hex_value(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

#[cfg(feature = "serde")]
struct HexVisitor<const N: usize> {
    type_name: &'static str,
}

#[cfg(feature = "serde")]
impl<'de, const N: usize> serde::de::Visitor<'de> for HexVisitor<N> {
    type Value = [u8; N];

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{} as hex or raw bytes", self.type_name)
    }

    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        parse_hex_array(value).map_err(E::custom)
    }

    fn visit_bytes<E>(self, value: &[u8]) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        bytes_from_slice(value).map_err(E::custom)
    }

    fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
    where
        A: serde::de::SeqAccess<'de>,
    {
        let mut bytes = [0u8; N];

        for (index, byte) in bytes.iter_mut().enumerate() {
            *byte = seq
                .next_element()?
                .ok_or_else(|| serde::de::Error::invalid_length(index, &self))?;
        }

        if seq.next_element::<u8>()?.is_some() {
            return Err(serde::de::Error::invalid_length(N + 1, &self));
        }

        Ok(bytes)
    }
}

#[cfg(feature = "serde")]
fn bytes_from_slice<const N: usize>(value: &[u8]) -> Result<[u8; N], HexParseError> {
    if value.len() != N {
        return Err(HexParseError::InvalidLength {
            expected: N,
            actual: value.len(),
        });
    }

    let mut bytes = [0u8; N];
    bytes.copy_from_slice(value);
    Ok(bytes)
}
