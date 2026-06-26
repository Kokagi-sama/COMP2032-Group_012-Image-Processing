import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

def save_output_image(output_image, output_path, input_image_name):
    # Create directory if it does not exist
    os.makedirs(output_path, exist_ok=True)
    # Save the images
    save_file_name = f'Output_{input_image_name}.png'
    save_file_path = os.path.join(output_path, save_file_name)

    cv2.imwrite(os.path.join(output_path, 'Grayscale_Image_' + save_file_name), output_image[0])
    cv2.imwrite(os.path.join(output_path, 'Bilateral_Filtered_Image_' + save_file_name), output_image[1])
    cv2.imwrite(os.path.join(output_path, 'Contrast_Stretched_Image_' + save_file_name), output_image[2])
    cv2.imwrite(os.path.join(output_path, 'Gamma_Corrected_Image_' + save_file_name), output_image[3])
    cv2.imwrite(os.path.join(output_path, 'Clahe_Image_' + save_file_name), output_image[4])
    cv2.imwrite(os.path.join(output_path, 'Gain_Image_' + save_file_name), output_image[5])
    cv2.imwrite(os.path.join(output_path, 'Multi_Level_Otsu_Thresholded_Image_' + save_file_name), output_image[6])
    cv2.imwrite(os.path.join(output_path, 'Closing_Image_' + save_file_name), output_image[7])
    cv2.imwrite(os.path.join(output_path, 'Contour_Filled_Image_' + save_file_name), output_image[8])
    cv2.imwrite(os.path.join(output_path, 'Largest_Contour_Image_' + save_file_name), output_image[9])
    cv2.imwrite(save_file_path, output_image[10])

    return save_file_path

def preprocess(image , inv):
    img = cv2.imread(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    final_img = None

    if (inv == True):
        
        _, final_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        _, final_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)

    return final_img

def calculate_bounding_box(image):
    # Define kernel for morphology operations
    kernel = np.ones((7, 7), np.uint8)

    # Use erosion and dilation
    eroded = cv2.erode(image, kernel, iterations=2)
    dilated = cv2.dilate(eroded, kernel, iterations=2)

    # Find contours on the morphologically processed image
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Create a black image of the same dimensions as the original
    black_background = np.zeros_like(image)

    # Sort the contours by area in descending order
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Check if we have at least one contour
    if sorted_contours:
        # Find the bounding box for the largest contour
        largest_contour = sorted_contours[0]
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Initialize combined bounding box as the largest one
        combined_x = x
        combined_y = y
        combined_x_max = x + w
        combined_y_max = y + h

        # Check if there's a second significant contour
        if len(sorted_contours) > 1 and cv2.contourArea(sorted_contours[1]) > 0.1 * cv2.contourArea(largest_contour):
            second_largest_contour = sorted_contours[1]
            x2, y2, w2, h2 = cv2.boundingRect(second_largest_contour)

            # Combine the bounding boxes
            combined_x = min(combined_x, x2)
            combined_y = min(combined_y, y2)
            combined_x_max = max(combined_x_max, x2 + w2)
            combined_y_max = max(combined_y_max, y2 + h2)

        # Draw all contours on the black background
        cv2.drawContours(black_background, sorted_contours, -1, (255, 255, 255), 1)
        
        # Draw the combined bounding box
        cv2.rectangle(black_background, (combined_x, combined_y), (combined_x_max, combined_y_max), (255, 255, 255), 2)

        # The combined bounding box coordinates
        bounding_box = [combined_x, combined_y, combined_x_max - combined_x, combined_y_max - combined_y]
    else:
        bounding_box = [0, 0, 0, 0]

    # Display the image with the largest contour and bounding box
    # cv2.imshow("Image with Largest Contour and Bounding Box", black_background)
    # cv2.waitKey(0)

    print(bounding_box)

    return bounding_box

def intersection_over_union(output_image_path, ground_truth_image_path):
    # Extract bounding boxes for both images
    output_image = preprocess(output_image_path, inv=False)
    inverted_ground_truth_image = preprocess(ground_truth_image_path, inv=True)
    

    boxA = calculate_bounding_box(inverted_ground_truth_image)
    boxB = calculate_bounding_box(output_image)

    # Calculate intersection rectangle coordinates
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Calculate area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # Calculate area of both bounding boxes
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # Calculate intersection over union (IoU)
    iou = interArea / float(boxAArea + boxBArea - interArea) if boxAArea + boxBArea - interArea > 0 else 0
    iou_percentage = iou * 100

    # Generate and save the comparison graph
    output_image_name = os.path.basename(output_image_path)
    graph_filename = f"Graph_{output_image_name.split('.')[0]}.png"
    graph_path = os.path.join(os.path.dirname(output_image_path), graph_filename)

    # Create the bar chart for IoU
    plt.bar(['IoU'], [iou_percentage])
    plt.ylabel('Accuracy (%)')
    plt.title('IoU Accuracy Comparison')
    plt.ylim(0, 100)  # Limit Y-axis to make graph clearer
    plt.savefig(graph_path)
    plt.close()

    return iou_percentage