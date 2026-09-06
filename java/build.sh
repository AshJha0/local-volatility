#!/usr/bin/env bash
# Build the localvol Java library, demo and tests into out/.
set -euo pipefail
cd "$(dirname "$0")"

JUNIT=/usr/share/java/junit4.jar
HAMCREST=/usr/share/java/hamcrest.jar

rm -rf out
mkdir -p out

# Library + demo (warnings are errors in spirit: -Xlint must stay clean).
javac -Xlint:all -d out $(find src/main/java -name '*.java')

# Tests (JUnit 4).
javac -Xlint:all -cp "out:${JUNIT}:${HAMCREST}" -d out $(find src/test/java -name '*.java')

echo "build OK"
