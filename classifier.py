import cv2 # computer vision library
import helpers # helper functions

import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg # for loading in images

# Using the load_dataset function in helpers.py
# Load training data
IMAGE_LIST = helpers.load_dataset("traffic_light_images")

######
#
# Preprocess Data
#
#####

# Function to standardize the rgb image input (resize to 32x32 pixel)
def standardize_input(image):
    standard_im = np.copy(image)
    standard_im = cv2.resize(image,(40,40))
    # Crop to remove surroundings
    standard_im = standard_im[4:-4, 4:-4, :]
    return standard_im

## Given a label - "red", "green", or "yellow" - return a one-hot encoded label
def one_hot_encode(label):
    one_hot_encoded = [] 
    if (label == "red"):
        one_hot_encoded = [1,0,0] 
    elif (label == "yellow"):
        one_hot_encoded = [0,1,0] 
    elif (label == "green"):
        one_hot_encoded = [0,0,1] 
    return one_hot_encoded

def standardize(image_list):
    # Empty image data array
    standard_list = []

    # Iterate through all the image-label pairs
    for item in image_list:
        image = item[0]
        label = item[1]

        # Standardize the image
        standardized_im = standardize_input(image)

        # One-hot encode the label
        one_hot_label = one_hot_encode(label)    

        # Append the image, and it's one hot encoded label to the full, processed list of image data 
        standard_list.append((standardized_im, one_hot_label))
        
    return standard_list

# Standardize all training images
STANDARDIZED_LIST = standardize(IMAGE_LIST)

#index = 0
#img = STANDARDIZED_LIST[index][0]
#plt.imshow(img)
#plt.show()
#print("Shape of the image: " + str(len(img)) + "x" + str(len(img[0])))
#print("Label of the image: " + str(STANDARDIZED_LIST[index][1]))

######
#
# Feature Extraction
#
#####

def normalize(feature):
    normalized = []
    normalizer = 32**2
    for i in range(len(feature)):
        normalized.append(feature[i] / normalizer)
    return normalized


def brightness_mask(rgb_image):
    """
    Function to mask (black out) dark regions
    """
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    
    # Mask image to remove brightness
    masked_image = np.copy(rgb_image)
    bright = np.array([0,0,200]) # optimized value
    brightest = np.array([225,255,255])
    
    mask = cv2.inRange(hsv, bright, brightest)
    
    masked_image[mask == 0] = [0, 0, 0]
    
    return masked_image

def calc_3zone_brightness(hsv_image):
    """
    Function to calculate the brightness sum of each third of a hsv image 
    """
    brightness = [0,0,0]
    s = hsv_image[:,:,1]
    
    for i in range(len(hsv_image)):
        if (i <= len(hsv_image) / 3):
            brightness[0] += sum(s[i])
        elif (i <= len(hsv_image)*2 / 3):
            brightness[1] += sum(s[i])
        else:
            brightness[2] += sum(s[i])

    return brightness

def get_brightness(rgb_image):
    """
    Function to create a brightness feature
    """
    feature = [0,0,0]
    
    masked_image = brightness_mask(rgb_image)
    #show_channels(cropped_image)
    
    # Detect average brightness of 3 sectors
    hsv = cv2.cvtColor(masked_image, cv2.COLOR_RGB2HSV)
    brightness = calc_3zone_brightness(hsv)

    return brightness

def get_value(rgb_image):
    """
    Function to create a value (hsv) feature
    """
    feature = [0,0,0]
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    v = hsv[:,:,2]

    for i in range(len(rgb_image)):
        if (i <= len(rgb_image) / 3):
            feature[0] += sum(v[i])
        elif (i <= len(rgb_image)*2 / 3):
            feature[1] += sum(v[i])
        else:
            feature[2] += sum(v[i])
    
    return normalize(feature)

def get_H_Values(rgb_image):
    """
    Function to create a hue (hsv) feature
    """
    H = [0,0,0]

    masked_image = brightness_mask(rgb_image)
    hsv = cv2.cvtColor(masked_image, cv2.COLOR_RGB2HSV)
    h = hsv[:,:,0]
    
    for i in range(len(rgb_image)): 
        if (i < len(rgb_image)/3):
            H[0] += sum(h[i])
        elif (i < len(rgb_image)*2/3):
            H[1] += sum(h[i])
        else:
            H[2] += sum(h[i])
    return normalize(H)

def ryg_values(rgb_image):
    """
    Function to create a red-yellow-green feature, summing the red / yellow / green value in the correspondent third
    """
    ryg = [0,0,0]
    
    masked_image = brightness_mask(rgb_image)    
    
    red_image = np.copy(masked_image)
    red_image[:,:,1] = 0
    red_image[:,:,2] = 0
    g_image = np.copy(masked_image)
    g_image[:,:,0] = 0
    g_image[:,:,2] = 0
    b_image = np.copy(masked_image)
    b_image[:,:,0] = 0
    b_image[:,:,1] = 0
    
    r = red_image[:,:,0]
    g = g_image[:,:,1]
    b = b_image[:,:,2]
    
    for i in range(len(rgb_image)): 
        if (i <= len(rgb_image)/3):
            ryg[0] += sum(r[i])
        elif (i < len(rgb_image)*2/3):
            ryg[1] += (sum(g[i]) + sum(r[i])) / 2
        else:
            ryg[2] += sum(g[i])
    
    return normalize(ryg)


######
#
# Classification
#
#####

def almost_same(val1, val2):
    """
    Determines if two values are almost the same ( +/- 1 )
    """
    return val1 == val2 or (val1-val2 < 3 and val1-val2 > -3)

def features_almost_same(feature):
    """
    Determines if features values are almost the same and thus invalid
    """
    if almost_same(feature[0], feature[1]):
        return True
    elif almost_same(feature[0], feature[2]):
        return True
    elif almost_same(feature[1], feature[2]):
        return True
    return False

def most_frequent(List): 
    """
    Returns the most frequent value in a list
    """
    return max(set(List), key = List.count)

def encode(estimate):
    """
    Encodes the estimate value to hot encoded output
    """
    predicted_label = []
    if estimate == 0:
        predicted_label = one_hot_encode("red")
    elif estimate == 1:
        predicted_label = one_hot_encode("yellow")
    elif estimate == 2:
        predicted_label = one_hot_encode("green")
    return predicted_label

def estimate_label(rgb_image):
    """
    This function determines the estimated label for a image by using all hsv layers.
    """
    predicted_label = []
    
    brightness = get_brightness(rgb_image)
    value = get_value(rgb_image)
    ryg = ryg_values(rgb_image)
    hue = get_H_Values(rgb_image)
    
    # Estimates
    bright_estimate = brightness.index(max(brightness))
    value_estimate = value.index(max(value))
    hue_estimate = hue.index(max(hue))
    ryg_estimate = ryg.index(max(ryg))
    
    estimates = []

    if not features_almost_same(brightness):
        estimates.append(bright_estimate)
    if not features_almost_same(value):
        estimates.append(value_estimate)
    if not features_almost_same(hue):
        estimates.append(hue_estimate)
    if not features_almost_same(ryg):
        estimates.append(ryg_estimate)

    if (len(estimates) > 0 and len(estimates) % 2 == 1):
        predicted_label = encode(most_frequent(estimates))
    elif not features_almost_same(ryg):
        predicted_label = encode(ryg_estimate)
    else:
        predicted_label = encode(value_estimate)

    return predicted_label   

random.shuffle(STANDARDIZED_LIST)


#####
#
# Test the algorithm
#
#####

# Constructs a list of misclassified images given a list of test images and their labels
def get_misclassified_images(test_images):
    misclassified_images_labels = []

    # Iterate through all the test images
    # Classify each image and compare to the true label
    for image in test_images:

        # Get true data
        im = image[0]
        true_label = image[1]

        # Get predicted label from your classifier
        predicted_label = estimate_label(im)

        # Compare true and predicted labels 
        if(predicted_label != true_label):
            misclassified_images_labels.append((im, predicted_label, true_label))
            
    # Return the list of misclassified [image, predicted_label, true_label] values
    return misclassified_images_labels


# Find all misclassified images in a given test set
MISCLASSIFIED = get_misclassified_images(STANDARDIZED_LIST)

# Accuracy calculations
total = len(STANDARDIZED_LIST)
num_correct = total - len(MISCLASSIFIED)
accuracy = num_correct/total

print('Accuracy: ' + str(accuracy))
print("Number of misclassified images = " + str(len(MISCLASSIFIED)) +' out of '+ str(total))

# Display misclassified
num = 5
false_image = MISCLASSIFIED[num][0]
plt.imshow(false_image)
print("Classified as: " + str(MISCLASSIFIED[num][1]))
print("Should be: " + str(MISCLASSIFIED[num][2]))
print("Brightness: " + str(get_brightness(false_image)))
print("Value: " + str(get_value(false_image)))
print("Hue: " + str(get_H_Values(false_image)))
print("Red/Yellow/Green: " + str(ryg_values(false_image)))
plt.show()