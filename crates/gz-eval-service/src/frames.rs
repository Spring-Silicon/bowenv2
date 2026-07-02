use crate::{ServiceError, ServiceResult};
use std::io::{Read, Write};
use std::os::unix::net::UnixStream;

pub const PROTOCOL_VERSION: u32 = 1;
pub const MAX_FRAME: usize = 256 * 1024 * 1024;

pub const FRAME_HELLO: u8 = 1;
pub const FRAME_HELLO_ACK: u8 = 2;
pub const FRAME_EVAL: u8 = 3;
pub const FRAME_EVAL_RESULT: u8 = 4;
pub const FRAME_PING: u8 = 5;
pub const FRAME_PONG: u8 = 6;
pub const FRAME_ERROR: u8 = 7;

pub fn read_frame<'a>(
    stream: &mut UnixStream,
    buf: &'a mut Vec<u8>,
) -> ServiceResult<(u8, &'a [u8])> {
    let mut len = [0u8; 4];
    stream
        .read_exact(&mut len)
        .map_err(|error| ServiceError::io(error.to_string()))?;
    let body_len = u32::from_le_bytes(len) as usize;
    if body_len == 0 {
        return Err(ServiceError::protocol("empty frame"));
    }
    if body_len > MAX_FRAME {
        return Err(ServiceError::protocol("frame exceeds maximum size"));
    }

    if buf.len() < body_len {
        buf.resize(body_len, 0);
    }
    stream
        .read_exact(&mut buf[..body_len])
        .map_err(|error| ServiceError::io(error.to_string()))?;
    let frame_type = buf[0];
    if !known_frame_type(frame_type) {
        return Err(ServiceError::protocol("unknown frame type"));
    }
    Ok((frame_type, &buf[1..body_len]))
}

pub fn write_frame(
    stream: &mut UnixStream,
    buf: &mut Vec<u8>,
    frame_type: u8,
    parts: &[&[u8]],
) -> ServiceResult<()> {
    if !known_frame_type(frame_type) {
        return Err(ServiceError::protocol("unknown frame type"));
    }
    let body_len = parts
        .iter()
        .try_fold(1usize, |total, part| total.checked_add(part.len()))
        .ok_or_else(|| ServiceError::protocol("frame length overflow"))?;
    if body_len > MAX_FRAME {
        return Err(ServiceError::protocol("frame exceeds maximum size"));
    }

    let frame_len = 4 + body_len;
    if buf.len() < frame_len {
        buf.resize(frame_len, 0);
    }
    buf[0..4].copy_from_slice(&(body_len as u32).to_le_bytes());
    buf[4] = frame_type;
    let mut cursor = 5;
    for part in parts {
        let end = cursor + part.len();
        buf[cursor..end].copy_from_slice(part);
        cursor = end;
    }
    stream
        .write_all(&buf[..frame_len])
        .map_err(|error| ServiceError::io(error.to_string()))
}

const fn known_frame_type(frame_type: u8) -> bool {
    matches!(
        frame_type,
        FRAME_HELLO
            | FRAME_HELLO_ACK
            | FRAME_EVAL
            | FRAME_EVAL_RESULT
            | FRAME_PING
            | FRAME_PONG
            | FRAME_ERROR
    )
}
