# Traffic Light Image Classifier (Python)
This classification algorithm was developed/coded during the "Into to Self-Driving Cars"-Nanodegree program.

The task was to implement a traffic light classification algorithm, that sorts RGB-Images of traffic lights into "red", "yellow" and green.

This includes:
* preprocessing the images (resize, crop) for standard input
* encoding the output into binary data (one hot encode)
* feature extraction from the images
    * masked surrounding
    * brightness feature (saturation from HSV image)
    * value feature (from HSV image)
    * hue feature (from HSV image)
    * red, yellow, green feature (summing red / yellow / green value in the correspondent third)
* classification

## How to run the code?

  1. Download the files;
  2. Run classifier.py in terminal
  3. See the results (accuracy and one missclassified image)

The accuracy should be over 95% and no red traffic light classified as green.

## Images
All images come from this [MIT self-driving car course](https://selfdrivingcars.mit.edu/) and are licensed under a [Creative Commons Attribution-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-sa/4.0/).
