#!/usr/bin/env bash
# Run the localvol JUnit 4 test suite (build.sh must have been run first).
set -euo pipefail
cd "$(dirname "$0")"

JUNIT=/usr/share/java/junit4.jar
HAMCREST=/usr/share/java/hamcrest.jar

java -cp "out:${JUNIT}:${HAMCREST}" org.junit.runner.JUnitCore \
  com.quant.localvol.TridiagTest \
  com.quant.localvol.SurfaceTest \
  com.quant.localvol.BlackScholesTest \
  com.quant.localvol.DupireTest \
  com.quant.localvol.PdeTest \
  com.quant.localvol.McTest \
  com.quant.localvol.RoundTripTest \
  com.quant.localvol.GoldenTest
