"""Allow `python -m media_filename_parser` execution."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
