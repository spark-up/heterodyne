#! /bin/bash

scripts="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

extra_usage() {
	echo \
		'To specify the path to the OpenStack administrative SSH key,' \
		'set the ADMIN_KEY environment variable.'
	echo \
		'This ADMIN_KEY should correspond to the OpenStack key with the name "admin".'
	echo
	echo -n \
		'To speed up provisioning, set the'
	printf ' %s,' NO_CREATE NO_PREPARE NO_DEPLOY_SPARK
	printf ' and'
	printf ' %s,' NO_DEPLOY_JUPYTER
	echo \
		' environment variables to 1.'
	echo
	echo \
		'Example:'
	printf '\t'
	echo \
		"NO_CREATE=1 NO_PREPARE=1 NO_DEPLOY_SPARK=1 $0 ..."
}

source "$scripts/_common.sh"

ssh_extra_args=()
if test -n "$ADMIN_KEY"; then
	ssh_extra_args+=(-i "$ADMIN_KEY")
fi

# shellcheck disable=SC2087
exec ssh "${ssh_extra_args[@]}" "$@" /bin/bash <<EOF
set -euo pipefail

cd ~/spark-openstack >/dev/null || exit 2

if ! test \
	-a ./spark-openstack \
	-a -a .venv/bin/activate \
; then
	echo 'Virtual environment or entrypoint not found' >&2
	exit 3
fi

source .venv/bin/activate

args=(
	./spark-openstack
)


# Negatives (Default to including flag)
test -z '$NO_CREATE' -a -z '$SKIP_CREATE' && args+=(--create)
test -z '$NO_DEPLOY_SPARK' && args+=(--deploy-spark)
test -z '$NO_DEPLOY_JUPYTER' && args+=(--deploy-jupyter)

# Affirmatives (Default to omitting flag)
test -n '$NO_PREPARE' -o -n '$SKIP_PREPARE' && args+=(--skip-prepare)

args+=(
		-k admin
		-s '${SPARK_WORKERS:-1}'
		-n flat-lan-1-net
		-t spark.custom
		-m spark.custom
		-a focal-server-cloudimg-amd64 
		--spark-version 2.4.8
		launch test-project
)
echo "\${args[@]}"

test -z '$DRY_RUN' || exit 0

"\${args[@]}"
EOF
