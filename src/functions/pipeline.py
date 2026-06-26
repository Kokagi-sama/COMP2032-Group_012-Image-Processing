import cv2
import numpy as np

# Preprocess Image functions

def bilateral_filter(image, diameter, sigmaColor, sigmaSpace):
    """
    Apply bilateral filtering.

    Reduce noise while preserving edges with bilateral filter.

    Parameters:
        image (numpy.ndarray): Input image.
        diameter (int): Diameter of each pixel neighbourhood.
        sigmaColor (float): Filter sigma in the color space.
        sigmaSpace (float): Filter sigma in the coordinate space.
        
    Returns:
        numpy.ndarray: Image after applying bilateral filtering.
    """
    return cv2.bilateralFilter(image, diameter, sigmaColor, sigmaSpace)


# Process image functions

def contrast_stretch(image):
    """
    Apply contrast stretching.

    Contrast stretching expands the range of pixel intensities in an image to cover the full dynamic range (0 to 255).

    Parameters:
        image (numpy.ndarray): Input image.

    Returns:
        numpy.ndarray: Image after applying contrast stretching.
    """
    # Convert to float to avoid clipping values
    img_float = image.astype(np.float32)
    
    # Get the minimum and maximum pixel values
    min_val = np.min(img_float)
    max_val = np.max(img_float)
    
    # Perform contrast stretching
    stretched_image = (img_float - min_val) / (max_val - min_val) * 255
    
    # Convert back to original data type
    stretched_image = np.clip(stretched_image, 0, 255).astype(np.uint8)
    
    return stretched_image

def gamma_correction(image, gamma_value):
    """
    Apply gamma correction.

    Gamma correction adjusts the brightness of the image by applying a power-law
    transformation to the pixel intensities.

    Parameters:
        image (numpy.ndarray): Input image.
        gamma_value (float): Gamma value for correction. Gamma value > 1 brightens the image, 
                             while gamma value < 1 darkens the image.
    
    Returns:
        numpy.ndarray: Image after applying gamma correction.
    """
    # Build a lookup table mapping pixel values [0, 255] to their adjusted gamma values
    inv_gamma = 1.0 / gamma_value
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
    
    # Apply gamma correction using the lookup table
    return cv2.LUT(image, table)

def clahe(image, clip_limit, tile_grid_size):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).

    CLAHE enhances the contrast of an image by applying histogram equalization locally in small regions.

    Parameters:
        image (numpy.ndarray): Input image.
        clip_limit (float): Threshold for contrast limiting. Higher values give more contrast.
        tile_grid_size (tuple): Size of grid for histogram equalization. (e.g., (8, 8))
    
    Returns:
        numpy.ndarray: Image after applying CLAHE.
    """
    # Create a CLAHE object
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    # Apply CLAHE
    return clahe.apply(image)

def gain(image, gain_factor):
    """
    Apply gain to an image by multiplying each pixel by a gain factor.

    Gain adjustment increases or decreases the brightness of an image by scaling pixel values.

    Parameters:
        image (numpy.ndarray): Input image.
        gain_factor (float): Factor to multiply each pixel value by. 
                             Values greater than 1 increase brightness, 
                             while values less than 1 decrease brightness.
    
    Returns:
        numpy.ndarray: Image after applying gain.
    """
    # Multiply each pixel by the gain factor
    return cv2.multiply(image, np.array([gain_factor], dtype=np.float32))

def multi_level_otsu_threshold(image, levels):
    """
    Apply Multi-Level Otsu Thresholding to segment an image into multiple levels.
    
    Multi-Level Otsu Thresholding segments an image into multiple levels based on the thresholds obtained using Otsu's method.

    Parameters:
        image (numpy.ndarray): Input image.
        levels (int): The number of thresholds to find, resulting in levels+1 segments.

    Returns:
        numpy.ndarray: Image after applying Multi-Level Otsu Thresholding.
    """
    # Compute the global threshold using Otsu's method to get the first level
    thresholds = [0]  # Start with 0 for the first threshold
    for i in range(levels):
        # Determine the thresholds
        _, thresh = cv2.threshold(image, thresholds[-1], 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        unique_values = np.unique(image[image > thresholds[-1]])
        if len(unique_values) > 0:
            next_thresh = np.mean(unique_values)
            thresholds.append(next_thresh)

    # Create a segmented image based on the thresholds
    segmented_image = np.zeros_like(image)
    for i in range(1, len(thresholds)):
        segmented_image[np.logical_and(image > thresholds[i-1], image <= thresholds[i])] = i * (255 // levels)

    # Handle the upper boundary
    segmented_image[image > thresholds[-1]] = 255

    return segmented_image


# Post processing functions

def closing(image, kernel_size):
    """
    Apply morphological closing to clean unwanted internal details.

    Morphological closing is a combination of dilation followed by erosion.
    It helps to close gaps between foreground pixels and smooth edges while removing small details.

    Parameters:
        image (numpy.ndarray): Input image.
        kernel_size (int): Size of the kernel used for morphological operations.

    Returns:
        np.ndarray: Image after applying closing.
    """
    # Define the structuring element for morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    # Dilate the image to close gaps between foreground pixels
    dilated_image = cv2.dilate(image, kernel, iterations=14)

    # Apply erosion to smooth edges and further remove small details
    eroded_image = cv2.erode(dilated_image, kernel, iterations=10)

    return eroded_image

def contour_fill(image):
    """
    Apply contour fill on a grayscale image.

    Fill the gaps within contours in the grayscale image.

    Parameters:
        image (numpy.ndarray): Input image.

    Returns:
        numpy.ndarray: Image after applying contour fill.
    """
    # Threshold the image to create a binary image
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on the image and fill them
    filled_image = np.zeros_like(image)
    cv2.drawContours(filled_image, contours, -1, (255), thickness=cv2.FILLED)

    return filled_image

def largest_contour(image):
    """
    Find and draw the largest contour.

    Parameters:
        image (numpy.ndarray): Input image.

    Returns:
        numpy.ndarray: Image with the largest contour filled.
    """
    # Find contours
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create an empty image to draw the largest contour
    largest_contour_image = np.zeros_like(image)
    
    # Check if contours were found
    if contours:
        # Find the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Draw the largest contour on the image
        cv2.drawContours(largest_contour_image, [largest_contour], -1, (255), thickness=cv2.FILLED)

    return largest_contour_image

def opening(image, kernel_size):
    """
    Apply contour fill.

    Fill the gaps within contours in the grayscale image.

    Parameters:
        image (numpy.ndarray): Input image.

    Returns:
        numpy.ndarray: Image after applying opening.
    """
    # Define kernel for morphology operations
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Use erosion and dilation
    eroded = cv2.erode(image, kernel, iterations=2)
    dilated = cv2.dilate(eroded, kernel, iterations=2)

    return dilated

# Pipeline for flower segmentation
def process_image(image, sigma):
    """
    Full pipeline for processing the image

    Parameters:
        image (numpy.ndarray): Input image.
        sigma: Array of sigma values

    Returns:
        numpy.ndarray: Binary image after processing through the pipeline.
    """
    # Convert image to grayscale if it is not already
    if len(image.shape) == 3:
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        grayscale_image = image

    # Pre-processing
    bilateral_filtered_image = bilateral_filter(grayscale_image, diameter=60, sigmaColor=sigma, sigmaSpace=90)

    # Process image
    contrast_stretched_image = contrast_stretch(bilateral_filtered_image)
    gamma_corrected_image = gamma_correction(contrast_stretched_image, gamma_value=0.18)
    clahe_image = clahe(gamma_corrected_image, clip_limit=1.5, tile_grid_size=(8,8))
    gain_image = gain(clahe_image, gain_factor=4.5)
    multi_level_otsu_thresholded_image = multi_level_otsu_threshold(gain_image, levels=3)
    
    # Post processing
    closed_image = closing(multi_level_otsu_thresholded_image, kernel_size=3)
    contour_filled_image = contour_fill(closed_image)
    largest_contour_image = largest_contour(contour_filled_image)
    opened_image = opening(largest_contour_image, kernel_size=7)


    return  grayscale_image, bilateral_filtered_image, contrast_stretched_image, gamma_corrected_image, clahe_image, gain_image, multi_level_otsu_thresholded_image, closed_image, contour_filled_image, largest_contour_image, opened_image 