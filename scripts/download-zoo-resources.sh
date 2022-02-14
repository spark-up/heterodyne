#! /bin/bash

set -eo pipefail

scripts="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
tmp="$(mktemp -d /tmp/gdown.XXXX)"
url='https://drive.google.com/drive/folders/1eC8F5pO2hSoQf4RQM7zww49y2ZbLIvqG?usp=sharing'

cd "$scripts" || exit 1

mkdir -p "$scripts/../data/zoo-resources"

if ! command gdown --version >/dev/null; then
	if ! test -a ".venv/bin/activate"; then
		python3 -m venv .venv
	fi

	# shellcheck disable=SC1091
	source .venv/bin/activate

	pip install -U gdown
fi

echo 'Downloading zoo resources...'

gdown --quiet --folder -O "$tmp" "$url"

mv "$tmp/MLFeatureTypeInference/resources/"* "$scripts/../data/zoo-resources"
rm -rf "$tmp"

echo '...Done!'
