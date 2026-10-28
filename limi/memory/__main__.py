"""CLI: `python -m limi.memory list` and `python -m limi.memory wipe`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from limi.config import read_db_path
from limi.memory.store import MemoryStore


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m limi.memory", description="Inspect or clear Limi's memory.")
    parser.add_argument("command", choices=["list", "wipe"])
    parser.add_argument("--db", help="database path (default: memory.db_path from config.yaml)")
    parser.add_argument("--config", help="config file to read the database path from")
    parser.add_argument("-y", "--yes", action="store_true", help="wipe without asking")
    args = parser.parse_args(argv)

    db_path = Path(args.db) if args.db else read_db_path(args.config)
    store = MemoryStore(db_path)
    try:
        if args.command == "list":
            facts = store.facts()
            print(f"Database: {db_path}")
            if facts:
                for fact in facts:
                    print(f"{fact.id:>3}. {fact.text}   ({fact.created_at})")
            else:
                print("No facts stored.")
            print(f"Dialog messages stored: {store.message_count()}")
            return 0

        if not args.yes:
            answer = input(f"Delete ALL facts and dialog history in {db_path}? Type 'yes' to confirm: ")
            if answer.strip().lower() != "yes":
                print("Cancelled, nothing deleted.")
                return 1
        store.wipe()
        print("Memory wiped.")
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    sys.exit(main())
