use crate::{PROTOCOL_VERSION, ServiceError, ServiceResult};
use gz_engine::{ActionSetHash, EngineId, EngineVersion, ModelVersion};
use gz_features::{BATCH_ENCODING_VERSION, FeatureSchemaHash};

pub const ERROR_PROTOCOL: u32 = 1;
pub const ERROR_ENCODING: u32 = 2;
pub const ERROR_SCHEMA: u32 = 3;
pub const ERROR_CAPACITY: u32 = 4;
pub const ERROR_MALFORMED: u32 = 5;

const HELLO_LEN: usize = 108;
const HELLO_ACK_LEN: usize = 28;
const ERROR_HEADER_LEN: usize = 6;
const MAX_ERROR_MESSAGE: usize = 512;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Hello {
    pub protocol_version: u32,
    pub encoding_version: u32,
    pub feature_schema_hash: FeatureSchemaHash,
    pub batch_capacity: u32,
    pub engine_id: EngineId,
    pub engine_version: EngineVersion,
    pub action_set_hash: ActionSetHash,
}

impl Hello {
    #[must_use]
    pub const fn new(
        feature_schema_hash: FeatureSchemaHash,
        batch_capacity: u32,
        engine_id: EngineId,
        engine_version: EngineVersion,
        action_set_hash: ActionSetHash,
    ) -> Self {
        Self {
            protocol_version: PROTOCOL_VERSION,
            encoding_version: BATCH_ENCODING_VERSION,
            feature_schema_hash,
            batch_capacity,
            engine_id,
            engine_version,
            action_set_hash,
        }
    }

    pub fn encode(&self, out: &mut Vec<u8>) {
        out.clear();
        out.extend_from_slice(&self.protocol_version.to_le_bytes());
        out.extend_from_slice(&self.encoding_version.to_le_bytes());
        out.extend_from_slice(self.feature_schema_hash.as_bytes());
        out.extend_from_slice(&self.batch_capacity.to_le_bytes());
        out.extend_from_slice(self.engine_id.as_bytes());
        out.extend_from_slice(self.engine_version.as_bytes());
        out.extend_from_slice(self.action_set_hash.as_bytes());
    }

    pub fn decode(bytes: &[u8]) -> ServiceResult<Self> {
        if bytes.len() != HELLO_LEN {
            return Err(ServiceError::handshake("bad HELLO length"));
        }
        Ok(Self {
            protocol_version: read_u32(bytes, 0)?,
            encoding_version: read_u32(bytes, 4)?,
            feature_schema_hash: FeatureSchemaHash::from_bytes(read_array(bytes, 8)?),
            batch_capacity: read_u32(bytes, 40)?,
            engine_id: EngineId::from_bytes(read_array(bytes, 44)?),
            engine_version: EngineVersion::from_bytes(read_array(bytes, 60)?),
            action_set_hash: ActionSetHash::from_bytes(read_array(bytes, 76)?),
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HelloAck {
    pub protocol_version: u32,
    pub model_version: ModelVersion,
    pub model_generation: u64,
}

impl HelloAck {
    pub fn encode(&self, out: &mut Vec<u8>) {
        out.clear();
        out.extend_from_slice(&self.protocol_version.to_le_bytes());
        out.extend_from_slice(self.model_version.as_bytes());
        out.extend_from_slice(&self.model_generation.to_le_bytes());
    }

    pub fn decode(bytes: &[u8]) -> ServiceResult<Self> {
        if bytes.len() != HELLO_ACK_LEN {
            return Err(ServiceError::handshake("bad HELLO_ACK length"));
        }
        Ok(Self {
            protocol_version: read_u32(bytes, 0)?,
            model_version: ModelVersion::from_bytes(read_array(bytes, 4)?),
            model_generation: u64::from_le_bytes(read_array(bytes, 20)?),
        })
    }
}

pub fn decode_error(bytes: &[u8]) -> ServiceResult<(u32, String)> {
    if bytes.len() < ERROR_HEADER_LEN {
        return Err(ServiceError::protocol("ERROR frame truncated"));
    }
    let code = read_u32(bytes, 0)?;
    let message_len = u16::from_le_bytes(
        bytes[4..6]
            .try_into()
            .expect("error message length slice checked"),
    ) as usize;
    if message_len > MAX_ERROR_MESSAGE || bytes.len() != ERROR_HEADER_LEN + message_len {
        return Err(ServiceError::protocol("bad ERROR frame length"));
    }
    let message = String::from_utf8_lossy(&bytes[ERROR_HEADER_LEN..]).into_owned();
    Ok((code, message))
}

fn read_u32(bytes: &[u8], offset: usize) -> ServiceResult<u32> {
    let slice = bytes
        .get(offset..offset + 4)
        .ok_or_else(|| ServiceError::protocol("u32 truncated"))?;
    Ok(u32::from_le_bytes(slice.try_into().expect("slice checked")))
}

fn read_array<const N: usize>(bytes: &[u8], offset: usize) -> ServiceResult<[u8; N]> {
    let slice = bytes
        .get(offset..offset + N)
        .ok_or_else(|| ServiceError::protocol("byte array truncated"))?;
    Ok(slice.try_into().expect("slice checked"))
}
