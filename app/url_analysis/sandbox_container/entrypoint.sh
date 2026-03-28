#!/bin/sh
set -eu

if [ "$#" -gt 0 ] && [ "$1" = "python" ]; then
  exec "$@"
fi

exec python /sandbox/sandbox_runner.py "$@"
