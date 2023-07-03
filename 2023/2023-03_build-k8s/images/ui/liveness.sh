#!/bin/bash

# Get timestamp: https://stackoverflow.com/a/12312982
echo "Hi from the liveness probe at $(date +%s)"

# Liveness output to pod logs: https://stackoverflow.com/a/75257695
echo "Hi from the liveness probe at $(date +%s)" > /proc/1/fd/1

python /src/probe.py
