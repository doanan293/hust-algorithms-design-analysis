import sys

from data.cli import main


raise SystemExit(main(["preprocess", *sys.argv[1:]]))
