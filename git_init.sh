#!/usr/bin/env bash

git init
# shellcheck disable=SC2035
git add *
git commit -m "init"

#git remote add origin git@github.com:yuanjie-ai/llm4gpt.git
#git branch -M master
git push -u origin master
# git remote remove origin
