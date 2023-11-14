#!/bin/bash
# FARLAB - UrbanECG 
# Developer: @mattwfranchi
# Last Edited: 11/14/2023 

# This script deletes all .out and .err files in all subdirectories of /log 

# Delete all .out files in /log
find ../log -name "*.out" -type f -delete

# Delete all .err files in /log
find ../log -name "*.err" -type f -delete

echo "Log files flushed."


