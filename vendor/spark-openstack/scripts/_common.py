import sys
from typing import NoReturn


def panic(message: str, status=1) -> NoReturn:
    print(message, file=sys.stderr)
    sys.exit(status)


def usage(code: int = 0) -> str:
    print('USAGE')
    print('\t%s <ssh-args ...>\n\n' % sys.argv[0])
    print('All arguments are passed through to ssh(1).')
    sys.exit(code)
