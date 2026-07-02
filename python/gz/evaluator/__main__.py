from __future__ import annotations

import argparse
import sys

from gz.common.log import setup
from gz.evaluator.backends import StubBackend
from gz.evaluator.server import serve


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", required=True)
    args = parser.parse_args(argv)
    log = setup("gz.evaluator")
    try:
        log.info("event=start socket=%s backend=stub", args.socket)
        serve(args.socket, StubBackend())
    except Exception as error:
        log.error("event=error error=%s", error)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
