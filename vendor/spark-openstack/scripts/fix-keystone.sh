#! /bin/bash

scripts="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd || exit 1 )"

source "$scripts/_common.sh"

PYTHONPATH=scripts/.. \
	python3 -m scripts.fix_keystone "$@"
