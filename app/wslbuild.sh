#!/bin/bash
# Build helper run inside Debian WSL: usage: bash wslbuild.sh <target-dir>
# Sets up the xenon env and runs make in the given directory.
set -u
export DEVKITXENON=/usr/local/xenon
export PATH="$PATH:$DEVKITXENON/bin:$DEVKITXENON/usr/bin"

cd "$1" || exit 1
make 2>&1
rc=$?
echo "MAKE_EXIT=$rc"
exit $rc
