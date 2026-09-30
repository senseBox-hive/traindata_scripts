#!/bin/bash

# Set the paths for Folder A and the three target folders
folder_a_path=$1
train_folder_path="./train"
val_folder_path="./val"
test_folder_path="./test"

# Create the target folders if they don't exist
mkdir -p "$train_folder_path"
mkdir -p "$val_folder_path"
mkdir -p "$test_folder_path"
mkdir -p "$train_folder_path/$folder_a_path"/*
mkdir -p "$val_folder_path/$folder_a_path"/*
mkdir -p "$test_folder_path/$folder_a_path"/*

# Clear the target folders if they are not empty
rm -rf "$train_folder_path/$folder_a_path"/*
rm -rf "$val_folder_path/$folder_a_path"/*
rm -rf "$test_folder_path/$folder_a_path"/*

# Set the split ratios (adjust these values as per your requirements)
train_ratio=0.7
val_ratio=0.2
test_ratio=0.1

# Get the total number of files in Folder A
total_files=$(ls "$folder_a_path" | wc -l)

# Calculate the number of files for each split
train_files=$(awk "BEGIN {printf \"%.0f\n\", $train_ratio * $total_files}")
val_files=$(awk "BEGIN {printf \"%.0f\n\", $val_ratio * $total_files}")
test_files=$(awk "BEGIN {printf \"%.0f\n\", $test_ratio * $total_files}")

# Move files from Folder A to the train folder
ls "$folder_a_path" | shuf -n $train_files | xargs -I {} mv "$folder_a_path/{}" "$train_folder_path/$folder_a_path/"

# Move files from Folder A to the val folder
ls "$folder_a_path" | shuf -n $val_files | xargs -I {} mv "$folder_a_path/{}" "$val_folder_path/$folder_a_path/"

# Move the remaining files from Folder A to the test folder
mv "$folder_a_path"/* "$test_folder_path/$folder_a_path"

echo "Files divided successfully!"
