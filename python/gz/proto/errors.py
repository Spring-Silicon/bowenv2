from __future__ import annotations

import struct

ERROR_PROTOCOL = 1
ERROR_ENCODING = 2
ERROR_SCHEMA = 3
ERROR_CAPACITY = 4
ERROR_MALFORMED = 5

_MAX_ERROR_MESSAGE = 512


class ProtocolError(Exception):
    def __init__(self, code: int, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(message)


def encode_error(code: int, message: str) -> bytes:
    encoded = message.encode("utf-8")[:_MAX_ERROR_MESSAGE]
    return struct.pack("<IH", code, len(encoded)) + encoded


def decode_error(buf: memoryview) -> tuple[int, str]:
    if len(buf) < 6:
        raise ProtocolError(ERROR_MALFORMED, "error frame truncated")
    code, msg_len = struct.unpack_from("<IH", buf, 0)
    if msg_len > _MAX_ERROR_MESSAGE or len(buf) != 6 + msg_len:
        raise ProtocolError(ERROR_MALFORMED, "bad error frame length")
    return code, bytes(buf[6:]).decode("utf-8", "replace")
