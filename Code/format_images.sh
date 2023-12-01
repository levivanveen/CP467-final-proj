#!/bin/bash

# Needed to convert HEIC files to PNG

# Change to the 'Scenes' directory
cd ../Scenes

# Rename files with .png suffix to .heic
for file in *.png; do
  mv "$file" "${file%.png}.heic"
done

# Convert HEIC files to PNG
for file in *.heic; do
  magick "$file" "${file%.heic}.png"
done

# Remove HEIC files
rm *.heic
