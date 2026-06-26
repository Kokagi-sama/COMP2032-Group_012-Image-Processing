import os
import cv2
import tkinter as tk
from tkinter import ttk
import shutil
from customtkinter import CTk, CTkFrame, CTkLabel, CTkButton, CTkCanvas, CTkProgressBar
from PIL import Image, ImageTk
from functions.pipeline import process_image
from functions.outputs import save_output_image, intersection_over_union

INPUT_PATH = './src/Dataset/'
OUTPUT_PATH = './src/output/'
SIGMA_VALUES = [*range(7, 8)]  # Example sigma range

class CustomTkinterApp:

    # Function to clear output directory at program start
    def clear_output_directory(self):
        if os.path.exists(OUTPUT_PATH):
            shutil.rmtree(OUTPUT_PATH)
        os.makedirs(OUTPUT_PATH, exist_ok=True)

    def __init__(self, master):
        self.master = master
        self.master.title("Image Processing App")

        # Set a fixed window size
        window_width = 1366
        window_height = 768
        self.master.geometry(f"{window_width}x{window_height}")

        # Get the screen dimension
        screen_width = self.master.winfo_screenwidth()
        screen_height = self.master.winfo_screenheight()

        # Calculate the position to center the window on the screen
        center_x = int((screen_width - window_width) / 2)
        center_y = int((screen_height - window_height) / 2)

        # Position the window at the center of the screen
        self.master.geometry(f'+{center_x}+{center_y}')

        # Initialise image filenames and current image index
        self.image_filenames = []
        self.current_image_index = 0

        # Initialise outputs generated
        self.outputs_exist = False

        # Initialise progress bar value
        self.progress_val = tk.DoubleVar()

        self.setup_start_screen()
        self.clear_output_directory()

    def setup_start_screen(self):
        self.start_frame = CTkFrame(self.master, fg_color="transparent")
        self.start_frame.grid(row=0, column=0, sticky="nsew")
        self.master.grid_rowconfigure(0, weight=1)
        self.master.grid_columnconfigure(0, weight=1)

        # Using an inner frame to center buttons
        button_frame = CTkFrame(self.start_frame, fg_color="transparent")
        button_frame.grid(row=0, column=0)
        self.start_frame.grid_rowconfigure(0, weight=1)
        self.start_frame.grid_columnconfigure(0, weight=1)

        # Configure button_frame for centering contents
        button_frame.grid_rowconfigure(0, weight=1)  # Add space at top
        button_frame.grid_rowconfigure(1, weight=1)  # Row for buttons with minimal vertical expansion
        button_frame.grid_rowconfigure(2, weight=1)  # Row for progress bar, also minimal vertical expansion
        button_frame.grid_rowconfigure(3, weight=1)  # Add space at bottom
        button_frame.grid_columnconfigure(0, weight=1)  # Ensure contents are centered horizontally
        
        # Initialize buttons and progress bar
        if self.outputs_exist:
            self.view_results_button = CTkButton(button_frame, text="View Results", command=self.view_results)
            self.view_results_button.grid(row=1, column=0, pady=10)
        else:
            self.generate_button = CTkButton(button_frame, text="Generate Outputs", command=self.generate_outputs)
            self.generate_button.grid(row=1, column=0, pady=10)
            self.progress_bar = CTkProgressBar(button_frame, variable=self.progress_val, mode='determinate')
            self.progress_bar.grid(row=2, column=0, pady=10)
            self.progress_bar.grid_remove() # Hide progress bar from view
            
        # Initialise exit button
        self.exit_button = CTkButton(button_frame, text="Exit", command=self.master.destroy)
        self.exit_button.grid(row=3, column=0, pady=10)  # Ensure it's also centered

    

    def get_images(self):
        images = []
        for subdir, dirs, files in os.walk(INPUT_PATH + 'input_images/'):
            for file in files:
                filepath = subdir + os.sep + file
                if filepath.endswith(".jpg") or filepath.endswith(".jpeg") or filepath.endswith(".png"):
                    images.append(filepath)
        return images

    def generate_outputs(self):
        if not self.outputs_exist:
            self.images = self.get_images()
            total_images = len(self.images)
            if total_images == 0:
                return
            
            self.generate_button.configure(state="disabled")
            self.progress_bar.grid()  # Show progress bar

            for index, image_path in enumerate(self.images):
                image = cv2.imread(image_path)
                input_image_name = os.path.splitext(os.path.basename(image_path))[0]
                difficulty_level = os.path.basename(os.path.dirname(image_path))
                output_path = os.path.join(OUTPUT_PATH, difficulty_level, input_image_name)
                ground_truth_path = os.path.join(INPUT_PATH, "ground_truths", difficulty_level, f"{input_image_name}.png")

                for sigma in SIGMA_VALUES:
                    final_image = process_image(image, sigma)
                    output_image_path = save_output_image(final_image, output_path, input_image_name)

                    # Making the graph image based on output image and ground truth
                    intersection_over_union(output_image_path, ground_truth_path)

                self.progress_val.set((index + 1) / total_images)
                self.master.update_idletasks()

            self.outputs_exist = True
            self.generate_button.configure(text="View Results", state="normal", command=self.view_results)
            self.progress_bar.grid_forget() 


    def view_results(self):
        self.start_frame.destroy()
        self.setup_results_screen()

    def setup_results_screen(self):
        self.results_frame = CTkFrame(self.master, fg_color="transparent")
        self.results_frame.grid(row=0, column=0)
        self.master.grid_rowconfigure(0, weight=1)
        self.master.grid_columnconfigure(0, weight=1)

        # Create an inner frame to hold content and center it
        content_frame = CTkFrame(self.results_frame, fg_color="transparent")
        content_frame.grid(row=0, column=0)
        content_frame.grid_columnconfigure(0, weight=1)
        content_frame.grid_columnconfigure(1, weight=1)
        content_frame.grid_columnconfigure(2, weight=1)
        content_frame.grid_columnconfigure(3, weight=1)
        content_frame.grid_rowconfigure(2, weight=1)

        self.image_canvases = {}
        for i, label in enumerate(["Input Image", "Output Image", "Ground Truth", "Graph"]):
            canvas = CTkCanvas(content_frame, bg="white", width=300, height=300)
            canvas.grid(row=0, column=i, padx=10, pady=10)
            canvas_label = CTkLabel(content_frame, text=label)
            canvas_label.grid(row=1, column=i)
            self.image_canvases[label] = canvas

        # Centering navigation controls
        self.pagination_frame = CTkFrame(content_frame, fg_color="transparent")
        self.pagination_frame.grid(row=2, column=0, columnspan=4)
        self.pagination_frame.grid_columnconfigure(0, weight=1)
        self.pagination_frame.grid_columnconfigure(1, weight=1)
        self.pagination_frame.grid_columnconfigure(2, weight=1)
        self.pagination_frame.grid_rowconfigure(0, weight=0, pad=10)  # Spacing for combobox row
        self.pagination_frame.grid_rowconfigure(1, weight=0, pad=10)  # Spacing for button row
        self.pagination_frame.grid_rowconfigure(2, weight=0, pad=10)  # Spacing for back button row

        # Labels and Comboboxes
        self.difficulty_label = CTkLabel(self.pagination_frame, text="Difficulty")
        self.difficulty_label.grid(row=0, column=1, padx=10 , pady=(30, 5))

        self.difficulty_selector = ttk.Combobox(self.pagination_frame, values=self.get_difficulties(), state="readonly")
        self.difficulty_selector.grid(row=0, column=2, padx=10, pady=(30, 5))
        self.difficulty_selector.bind("<<ComboboxSelected>>", self.difficulty_changed)
        self.difficulty_selector.set(self.get_difficulties()[0])

        self.output_label = CTkLabel(self.pagination_frame, text="Output Image")
        self.output_label.grid(row=0, column=3, padx=10, pady=(30, 5))

        self.output_selector = ttk.Combobox(self.pagination_frame, values=self.get_image_filenames("easy"), state="readonly")
        self.output_selector.grid(row=0, column=4, padx=10, pady=(30, 5))
        self.output_selector.bind("<<ComboboxSelected>>", self.update_view)
        self.output_selector.set(self.output_selector['values'][0] if self.output_selector['values'] else "No Image Available!")

        # Navigation Buttons
        self.prev_button = CTkButton(self.pagination_frame, text="Previous", command=self.previous_image)
        self.prev_button.grid(row=1, column=1, columnspan=2, padx=(80, 5))

        self.next_button = CTkButton(self.pagination_frame, text="Next", command=self.next_image)
        self.next_button.grid(row=1, column=3, columnspan=2, padx=(5, 80))

        # Back Button
        self.back_button = CTkButton(self.pagination_frame, text="Back", command=self.back_to_start)
        self.back_button.grid(row=2, column=1, columnspan=4)

        # Initialize output based on difficulty
        self.difficulty_changed()



    def get_difficulties(self):
        # Sort Order
        difficulty_order = {'easy': 1, 'medium': 2, 'hard': 3}
        
        # Get the list of directories in the output path
        difficulties = os.listdir(OUTPUT_PATH)
        
        # Sort the difficulties using the predefined order
        difficulties.sort(key=lambda x: difficulty_order.get(x, 0))
        
        return difficulties

    
    def display_image(self, path, canvas, image_type=""):
        canvas.delete("all")
        if os.path.exists(path) and os.path.isfile(path):
            img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)

            # Get original dimensions
            original_width, original_height = img.size

            # Calculate new dimensions
            new_width = int(original_width * 0.4)
            new_height = int(original_height * 0.4)

            # Resize image
            img = img.resize((new_width, new_height))
            
            img_tk = ImageTk.PhotoImage(image=img)
            canvas.create_image(150, 150, image=img_tk)
            canvas.image = img_tk  # Keep a reference to avoid garbage collection
        else:
            # If the path is a directory and we're looking for output or Graphs, we display the first image.
            if image_type in ["Output Image", "Graph"] and os.path.isdir(path):
                files = sorted(os.listdir(path))
                if files:
                    self.display_image(os.path.join(path, files[0]), canvas, image_type)
                else:
                    canvas.create_text(150, 150, text="No images found!", anchor="center")
            else:
                canvas.create_text(150, 150, text="Image not found!", anchor="center")

    def difficulty_changed(self, event=None):
        difficulty = self.difficulty_selector.get()
        self.image_filenames = self.get_image_filenames(difficulty)
        if self.image_filenames:
            self.current_image_index = 0
            self.output_selector['values'] = self.image_filenames
            self.output_selector.current(self.current_image_index)
            self.update_view()  # This will trigger the update for the initial view.
        else:
            # Handle the case where there are no images available for the selected difficulty.
            for canvas in self.image_canvases.values():
                canvas.delete("all")
                canvas.create_text(150, 150, text="No images available!", anchor="center")

                
    def get_image_filenames(self, difficulty):
        input_dir = os.path.join(INPUT_PATH, "input_images", difficulty)
        return [os.path.splitext(file)[0] for file in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, file))]

    def update_image_selectors(self):
        self.output_selector['values'] = self.image_filenames
        self.output_selector.set(self.image_filenames[self.current_image_index] if self.image_filenames else "")
        self.update_view()

    def update_view(self, event=None):  # Accept an optional event argument
        difficulty = self.difficulty_selector.get()
        selected_image_name = self.output_selector.get()
        if selected_image_name:
            self.current_image_index = self.image_filenames.index(selected_image_name)
        else:
            for canvas in self.image_canvases.values():
                canvas.delete("all")
                canvas.create_text(150, 150, text="No images available!", anchor="center")
            return

        image_subdir = self.image_filenames[self.current_image_index]
        input_image_path = os.path.join(INPUT_PATH, "input_images", difficulty, f"{image_subdir}.jpg")
        output_image_dir = os.path.join(OUTPUT_PATH, difficulty, image_subdir)
        ground_truth_path = os.path.join(INPUT_PATH, "ground_truths", difficulty, f"{image_subdir}.png")

        # Assuming the first image in the sorted list of output images is the one to display
        output_image_path = self.get_output_image(output_image_dir)

        graph_path = self.get_graph(output_image_dir)

        paths = {
            "Input Image": input_image_path,
            "Output Image": output_image_path,
            "Ground Truth": ground_truth_path,
            "Graph": graph_path
        }

        for label, path in paths.items():
            self.display_image(path, self.image_canvases[label], label)

        self.update_navigation_buttons()

    def get_output_image(self, output_image_dir):
        try:
            # List all files in the directory
            files = os.listdir(output_image_dir)
            
            # Filter files that start with "Output" case insensitive
            output_files = [file for file in files if file.lower().startswith("output")]
            
            # Sort the filtered files to maintain a consistent order
            output_files.sort()
            
            # Return the first file from the sorted list, joined with the full path
            return os.path.join(output_image_dir, output_files[0]) if output_files else ""
        except FileNotFoundError:
            return ""
        
    def get_graph(self, output_image_dir):
        try:
            # List all files in the directory
            files = os.listdir(output_image_dir)
            
            # Filter files that start with "Output" case insensitive
            output_files = [file for file in files if file.lower().startswith("graph")]
            
            # Sort the filtered files to maintain a consistent order
            output_files.sort()
            
            # Return the first file from the sorted list, joined with the full path
            return os.path.join(output_image_dir, output_files[0]) if output_files else ""
        except FileNotFoundError:
            return ""


    def next_image(self):
        if self.current_image_index < len(self.image_filenames) - 1:
            # Move to next image in current difficulty
            self.current_image_index += 1
        else:
            # Move to next difficulty if available
            difficulties = self.get_difficulties()
            current_difficulty_index = difficulties.index(self.difficulty_selector.get())
            if current_difficulty_index < len(difficulties) - 1:
                self.difficulty_selector.set(difficulties[current_difficulty_index + 1])
                self.difficulty_changed()
                self.current_image_index = 0
            else:
                # Disable Next button if no more difficulties
                self.next_button.configure(state="disabled")
                return
        self.update_image_selectors()
        self.update_view()

    def previous_image(self):
        if self.current_image_index > 0:
            # Move to previous image in current difficulty
            self.current_image_index -= 1
        else:
            # Move to previous difficulty if available
            difficulties = self.get_difficulties()
            current_difficulty_index = difficulties.index(self.difficulty_selector.get())
            if current_difficulty_index > 0:
                self.difficulty_selector.set(difficulties[current_difficulty_index - 1])
                self.difficulty_changed()
                self.current_image_index = len(self.image_filenames) - 1
            else:
                # Disable Previous button if at the beginning
                self.prev_button.configure(state="disabled")
                return
        self.update_image_selectors()
        self.update_view()

    def update_navigation_buttons(self):
        # Enable both buttons first
        self.next_button.configure(state="normal")
        self.prev_button.configure(state="normal")

        # Check if we are at the first image of the first difficulty
        if self.current_image_index == 0 and self.difficulty_selector.get() == self.get_difficulties()[0]:
            self.prev_button.configure(state="disabled")

        # Check if we are at the last image of the last difficulty
        difficulties = self.get_difficulties()
        if (self.current_image_index == len(self.image_filenames) - 1 and
           self.difficulty_selector.get() == difficulties[-1]):
            self.next_button.configure(state="disabled")
    
            
    def back_to_start(self):
        # Function to return to the starting screen
        self.results_frame.destroy()
        self.setup_start_screen()

    

def main():
    root = CTk()
    app = CustomTkinterApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()