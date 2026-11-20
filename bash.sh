#! /usr/bin/bash
# Source this file when working on this project
# so that I can use short cuts.


# envvars
export WORKDIR=$(pwd)
export TRASH="$WORKDIR/.trash"

# rip: rm alternative
# https://github.com/nivekuil/rip
export GRAVEYARD=$WORKDIR/.trash/


# timestamp: current time in a compact, sortable format
timestamp() {
  local format='%Y%m%d-%H%M%S'
  date "+$format"
}

# flatten repo for LLM context
flatten_repo() {
    # Help
    if [[ "${1:-}" == "-h" ]]; then
        echo "WARNING: if you are sending the function output to a file, make sure it is written OUTSIDE the repo (e.g. to ../out.txt) to prevent INFINITE LOOPS."
        return 0
    fi
  fd --type f --size -1m --print0 |
    xargs -0 -I {} sh -c '
      printf "\n--- %s ---\n" "$1"
      cat -- "$1"
      printf "\n"
    ' sh {}

}


# timestamp: current time in a compact, sortable format
timestamp() {
  local format='%Y%m%d-%H%M%S'
  date "+$format"
}
