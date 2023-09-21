#!/bin/bash
# Bash script to print the number of files in the current directory, including all subdirectories recursively, every 10 seconds.
# Usage: bash count_files.sh

while true
do
    find $1 -type f | wc -l
    sleep 10
done
