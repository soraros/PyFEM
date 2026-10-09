"""Allow ``python -m pyfem`` as an alias for the ``pyfem`` console script."""

import sys

from pyfem.core.cli import main

if __name__ == "__main__":
    main(sys.argv[1:])
