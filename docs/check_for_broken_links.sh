#!/usr/bin/env bash

DOCS_DIR=$(dirname "${BASH_SOURCE[0]}")
FALSE_POSITIVES_JSON="${DOCS_DIR}/false_positives.json"
LINKCHECK_JSON="${DOCS_DIR}/build/linkcheck/output.json"

function check_environment {
  local err=0
  if ! [ -x "$(command -v jq)" ]; then
    >&2 echo "jq is required but is not found."
    ((err++))
  fi
  if [ ! -f "${FALSE_POSITIVES_JSON}" ]; then
    >&2 echo "A JSON file with false positives is required: ${FALSE_POSITIVES_JSON}"
    ((err++))
  fi
  if [ ! -f "${LINKCHECK_JSON}" ]; then
    >&2 echo "Did not find linkcheck output JSON file: ${LINKCHECK_JSON}."
    >&2 echo "Run Sphinx with the linkcheck arg: make -C docs clean linkcheck"
    ((err++))
  fi
  if [ "${err}" -gt 0 ]; then
    exit 2
  fi
}

function check_links {
  local err=0
  # If you know how to prevent the hack with using jq twice, lmk.
  broken=$(jq 'select(.status == "broken")' "${LINKCHECK_JSON}" | jq -s)
  count=$(echo "${broken}" | jq 'length')
  for i in $(seq 0 $(($count - 1)))
  do
    entry=$(echo "${broken}" | jq ".[${i}]")
    link=$(echo "${entry}" | jq -r '.uri')
    [ -n "${DEBUG}" ] && {
     echo >&2 "Checking for false positive: ${link}"
    }
    local resp; resp=$(jq --arg check "${link}" -s 'any(.uri == $check)' < "${FALSE_POSITIVES_JSON}")
    # "false" indicates that the URL did not match any of the URIs in the false positive file.
    if [ "false" = "${resp}" ]; then
      ((err++))
      echo "${entry}"
    fi
  done
  exit "${err}"
}

check_environment
check_links
