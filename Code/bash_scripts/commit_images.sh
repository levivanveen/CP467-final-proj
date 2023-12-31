#!/bin/bash

# Get the list of modified files
modified_files=$(git status -s | awk '{print $2}')

# Set the initial commit number
commit_number=1

# Commit and push 4 files at a time
for file in $modified_files; do
    git add $file
    count=$((count + 1))

    # Commit and push every 4 files
    if [ $count -eq 4 ]; then
        git commit -m "Pushing all output images $commit_number"
        git push origin main
        wait
        count=0
        commit_number=$((commit_number + 1))
    fi
done
