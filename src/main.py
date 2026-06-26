import shutil
import os
import cv2
from functions.pipeline import process_image
from functions.outputs import save_output_image

# Global variables for input and output paths
INPUT_PATH = './src/Dataset/'
OUTPUT_PATH = './src/output/'

def get_images():
    images = []
    for subdir, dirs, files in os.walk(INPUT_PATH + 'input_images/'):
        for file in files:
            filepath = subdir + os.sep + file
            if filepath.endswith(".jpg") or filepath.endswith(".jpeg") or filepath.endswith(".png"):
                images.append(filepath)
    return images

# Ensuring the directory for outputs is cleared before running the program
def clear_output_directory(path=OUTPUT_PATH):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)

def main():
    try:
        # Clear output directory at the start
        clear_output_directory(OUTPUT_PATH)
        images = get_images()
        print(images)
        sigma_values = [7]
        
        print("Generating output images....!")
        
        for image_path in images:
            image = cv2.imread(image_path)
            input_image_name = os.path.splitext(os.path.basename(image_path))[0]
            difficulty_level = os.path.basename(os.path.dirname(image_path))
        
            output_path = os.path.join(OUTPUT_PATH, difficulty_level, input_image_name)
            for sigma in sigma_values:
                final_image = process_image(image, sigma)
                save_output_image(final_image, output_path, sigma, input_image_name)
            
        print("Output images generated successfully!")

    except Exception as e:
        print(f"An error occurred: {e}")
        
if __name__ == "__main__":
    main()