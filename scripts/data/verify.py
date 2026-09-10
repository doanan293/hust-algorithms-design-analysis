import sys

from data.cli import main


raise SystemExit(main(["verify", *sys.argv[1:]]))
