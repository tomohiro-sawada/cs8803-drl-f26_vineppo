#! /usr/bin/bash
# Source this file when working on this project
# so that I can use short cuts.


# envvars
export WORKDIR=$(pwd)
export TRASH="$WORKDIR/.trash"

# rip: rm alternative
# https://github.com/nivekuil/rip
export GRAVEYARD=$WORKDIR/.trash/



# flatten repo: useful for prompting ChatGPT
flatten_repo() {
  fd --type f --size -1m --print0 |
    xargs -0 -I {} sh -c '
      printf "\n--- %s ---\n" "$1"
      cat -- "$1"
      printf "\n"
    ' sh {}
}
