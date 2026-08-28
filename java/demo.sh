#!/usr/bin/env bash
# Run the localvol demo (build.sh must have been run first).
set -euo pipefail
cd "$(dirname "$0")"
java -cp out com.quant.localvol.Demo "${1:-../data}"
