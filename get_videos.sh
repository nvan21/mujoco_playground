#!/usr/bin/env bash

set -euo pipefail

usage() {
    echo "Usage: $0 {batiquitos|cowles|volcan}"
    exit 1
}

if [[ $# -ne 1 ]]; then
    usage
fi

case "$1" in
    batiquitos|cowles|volcan)
        host="nathan@${1}.ucsd.edu"
        ;;
    *)
        echo "Error: unknown server '${1}'"
        usage
        ;;
esac

mkdir -p videos

echo "Syncing mp4 files from ${host}:/data/nathan/mujoco_playground/ -> ./videos/"
rsync -avz --include="*.mp4" --include="*/" --exclude="*" \
    "${host}:/data/nathan/mujoco_playground/" ./videos/
