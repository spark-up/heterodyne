#! /bin/bash

scripts="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

source "$scripts/_common.sh"

# shellcheck disable=SC2087
exec ssh "$@" /bin/bash <<EOF
cd ~/spark-openstack || exit 2

git pull
EOF
