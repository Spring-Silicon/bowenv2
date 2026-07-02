use std::fmt;

pub type ServiceResult<T> = Result<T, ServiceError>;

const MAX_MESSAGE_BYTES: usize = 512;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ServiceError {
    Handshake(String),
    Protocol(String),
    Backend { code: u32, message: String },
    Io(String),
}

impl ServiceError {
    pub fn handshake(message: impl AsRef<str>) -> Self {
        Self::Handshake(bound_message(message.as_ref()))
    }

    pub fn protocol(message: impl AsRef<str>) -> Self {
        Self::Protocol(bound_message(message.as_ref()))
    }

    pub fn backend(code: u32, message: impl AsRef<str>) -> Self {
        Self::Backend {
            code,
            message: bound_message(message.as_ref()),
        }
    }

    pub fn io(message: impl AsRef<str>) -> Self {
        Self::Io(bound_message(message.as_ref()))
    }
}

impl fmt::Display for ServiceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Handshake(message) => write!(f, "handshake failed: {message}"),
            Self::Protocol(message) => write!(f, "protocol error: {message}"),
            Self::Backend { code, message } => {
                write!(f, "backend error {code}: {message}")
            }
            Self::Io(message) => write!(f, "eval service io error: {message}"),
        }
    }
}

impl std::error::Error for ServiceError {}

pub(crate) fn bound_message(message: &str) -> String {
    if message.len() <= MAX_MESSAGE_BYTES {
        return message.to_owned();
    }

    let mut end = 0;
    for (index, ch) in message.char_indices() {
        let next = index + ch.len_utf8();
        if next > MAX_MESSAGE_BYTES {
            break;
        }
        end = next;
    }
    message[..end].to_owned()
}
