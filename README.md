# bee data scripts
scripts for preparing the bee training data.
see https://github.com/senseBox-hive/hive-observer-firmware for main project.
the algorithm used in the crop script should ideally be identical to the one used in practice by the bee sensor.

## crop.py
uses the CCL algorithm to extract and save candidate crops from the frames.
Example: `python3 crop.py --input-dir uncropped --output-dir unsorted`

## remove_dupes.py
crop.py sometimes creates crops in which large sections overlap with other crops. To prevent these sections from showing up in the training data multiple times, or even be duplicated in training and validation, this script attempts to remove overlapping crops while still keeping the amount of crops as high as possible.
Example: `python3 remove_dupes.py --work-dir unsorted --threshold 0.25 && rm -rf unsorted/to_delete`

## sort.py
Program for labeling the classes of the crops. Use the number keys to assign a class to the crop shown.
Example: `python3 sort.py --input-dir unsorted`

## convert_to_raw565.py
Converts an image into RGB565 pixel data as used by the ESP32. Used for testing.
Example: `python3 convert_to_raw565.py example.png example.rgb565`