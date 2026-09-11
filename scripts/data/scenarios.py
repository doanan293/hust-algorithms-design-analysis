import sys

from data.cli import main


raise SystemExit(main(["scenarios", *sys.argv[1:]]))
