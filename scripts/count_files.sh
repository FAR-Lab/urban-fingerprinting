#!/bin/bash
# Bash script to print the number of files in the current directory, including all subdirectories recursively, every 10 seconds.
# Usage: bash count_files.sh

while true
do
    sum=0
    for i in {0..63}
    do
        directory="$1/$i"
        if [ -d "$directory" ]; then
            num=$(find "$directory" -type f | wc -l)
            sum=$((sum + num))
        fi
    done
    echo $sum
    sleep 60
done