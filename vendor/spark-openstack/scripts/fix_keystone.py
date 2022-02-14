#! /usr/bin/env python3

import os
import shlex
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

from ._common import panic, usage

SCRIPTS = Path(__file__).parent.resolve()


def main():
    if len(sys.argv) < 2:
        usage()

    host = os.getenv('API_HOST', None)
    if not host:
        host = input('API_HOST: ')

    if not host.startswith('http://') and not host.startswith('https://'):
        host = 'http://' + host

    parse = urlparse(host)
    if parse.path or parse.params or parse.query or parse.fragment:
        panic('API_HOST must only contain hostname and optional scheme.')
    if ':' in parse.netloc:
        panic('API_HOST must not include port.')

    script = f'API_HOST={shlex.quote(host)}\n'
    script += Path(SCRIPTS / '_fix-keystone.sh').read_text()

    args = ['ssh', '-T']
    args.extend(sys.argv[1:])
    proc = subprocess.run(args, input=script, encoding='utf8')
    exit(proc.returncode)


if __name__ == '__main__':
    main()
