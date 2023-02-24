#!/usr/bin/env bash

__dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
__root="$(cd "$(dirname "${__dir}")" && pwd)"

pip install pkginfo twine

pkginfo -f requires_dist ${__root}/dist/*.whl | grep "@"

if [ $? -eq 0 ]; then
    cat <<EOF
::error::Found direct reference(s) in package requires_dist. Please use only normal version specifiers.
See https://peps.python.org/pep-0440/#direct-references for more details.
EOF
    exit 1
fi

twine check ${__root}/dist/*
