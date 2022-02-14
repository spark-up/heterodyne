#! /bin/bash

usage() {
	echo 'USAGE'
	printf '\t%s <ssh-args ...>\n\n' "$0"
	echo 'All arguments are passed through to ssh(1).'
	declare -F extra_usage >/dev/null && echo && extra_usage
}

if [[ "$#" -lt 1 ]]; then
	usage >&2
	exit 1
fi

_check_help() {
	while [[ $# -gt 0 ]]; do
		case "$1" in
			-h|--help)
				usage >&2
				exit 0
				;;
			--)
				return 0
				;;
			*)
				shift
				;;
		esac
	done
}

_check_help "$@"
