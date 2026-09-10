import sys

from data.cli import main


raise SystemExit(main(["download", *sys.argv[1:]]))
