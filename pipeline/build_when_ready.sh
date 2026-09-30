#!/usr/bin/env bash
# Waits for download.sh to get past pagelinks, then runs build_core. Run from anywhere:
#   setsid nohup pipeline/build_when_ready.sh > data/build.log 2>&1 &
cd "$(dirname "$0")/.."
log=${WIKI_DUMPS:-dumps}/download.log
until grep -q 'downloading categorylinks' "$log" || grep -q '\] done' "$log"; do sleep 60; done
python3 -m pipeline.build_core
