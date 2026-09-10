import sys

from data.cli import main


raise SystemExit(main(["inventory", *sys.argv[1:]]))
