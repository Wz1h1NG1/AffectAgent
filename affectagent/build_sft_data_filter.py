"""Filter SFT records only; see :mod:`affectagent.build_sft_data`."""

import sys

from .build_sft_data import main


if __name__ == "__main__":
    sys.argv.extend(["--roles", "filter"])
    main()
