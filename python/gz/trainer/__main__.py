from __future__ import annotations

import argparse
import sys

from gz.common.log import setup
from gz.trainer.driver import run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    log = setup("gz.trainer")
    try:
        log.info("event=start config=%s", args.config)
        run(args.config)
    except Exception as error:
        log.error("event=error error=%s", error)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
