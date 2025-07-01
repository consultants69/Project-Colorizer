import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os

# --- Model Paths ---
PROTOTXT_PATH = 'model/colorization_deploy_v2.prototxt'
MODEL_PATH = 'model/colorization_release_v2.caffemodel'
POINTS_PATH = 'model/pts_in_hull.npy'

# --- Global variable for the loaded network ---
net = None  # Holds loaded deep learning network object
points = None   # Holds the cluster centers loaded from .npy file. 

def load_colorization_model():
    """Loading the pre-trained colorization model and cluster centers."""
    global net, points
    if net is None:
        try:
            print("[INFO] Loading colorization model...")
            # Load the network architecture and weights
            net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, MODEL_PATH)
            #Load the cluster centers using numpy
            points = np.load(POINTS_PATH)

            # Add the cluster centers as 1x1 convolutions to the model
            pts_in_hull = points.transpose().reshape(2, 313, 1, 1)
            net.getLayer(net.getLayerId('class8_ab')).blobs = [pts_in_hull.astype("float32")]
            net.getLayer(net.getLayerId('conv8_313_rh')).blobs = [np.full([1, 313], 2.606, dtype="float32")]
            print("[INFO] Model loaded successfully.")
            return True
        except Exception as e:
            print(f"[ERROR] Could not load model: {e}")
            messagebox.showerror("Error", f"Could not load the colorization model.\nMake sure 'model' folder with the files is in the same directory.\nDetails: {e}")
            net = None # Ensure net is None if loading fails
            points = None
            return False
    return True # Model is already loaded

def colorize_image(image_path, output_path):
    """Colorizes a single black and white image."""
    if not load_colorization_model():
        return False

    try:
        # Load the input image
        image = cv2.imread(image_path)
        if image is None:
            messagebox.showerror("Error", f"Could not read image file: {image_path}")
            return False

        # Convert to Lab color space
        scaled = image.astype("float32") / 255.0
        lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

        # Resize the image and split channels
        resized = cv2.resize(lab, (224, 224))
        L = cv2.split(resized)[0]
        L -= 50 # Subtract mean

        # Perform inference
        net.setInput(cv2.dnn.blobFromImage(L))
        ab = net.forward()[0, :, :, :].transpose((1, 2, 0))

        # Resize 'ab' channels back to original image size
        ab = cv2.resize(ab, (image.shape[1], image.shape[0]))

        # Get the original L channel and combine with predicted 'ab'
        L_orig = cv2.split(lab)[0]
        colorized = np.concatenate((L_orig[:, :, np.newaxis], ab), axis=2)

        # Convert back to BGR and save
        colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
        colorized = np.clip(colorized, 0, 1) # Clip values before converting to uint8
        colorized = (255 * colorized).astype("uint8")

        cv2.imwrite(output_path, colorized)
        print(f"[INFO] Image colorized and saved to {output_path}")
        return True

    except Exception as e:
        print(f"[ERROR] Error during image colorization: {e}")
        messagebox.showerror("Error", f"An error occurred during image colorization:\n{e}")
        return False

def colorize_video(video_path, output_path, progress_callback=None):
    """Colorizes a black and white video frame by frame."""
    if not load_colorization_model():
        return False

    try:
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            messagebox.showerror("Error", f"Could not open video file: {video_path}")
            return False

        # Get video properties
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Define the codec and create VideoWriter object
        # 'mp4v' or 'xvid' codecs are generally compatible
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Or 'XVID'
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        if not out.isOpened():
             messagebox.showerror("Error", f"Could not create output video file: {output_path}\nMake sure the output file path is valid and the codec is supported.")
             cap.release()
             return False

        print(f"[INFO] Colorizing video: {video_path} ({frame_count} frames at {fps} fps)")

        for i in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                print(f"[INFO] Could not read frame {i}, stopping.")
                break

            # Ensure the frame is treated as grayscale input (convert to BGR first for Lab conversion)
            if len(frame.shape) == 2: # If it's already grayscale
                 frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            else: # If it has 3 channels, assume it's BW in BGR or similar
                 frame_bgr = frame

            # Convert to Lab color space
            scaled = frame_bgr.astype("float32") / 255.0
            lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

            # Resize the image and split channels (for model input)
            resized = cv2.resize(lab, (224, 224))
            L = cv2.split(resized)[0]
            L -= 50 # Subtract mean

            # Perform inference
            net.setInput(cv2.dnn.blobFromImage(L))
            ab = net.forward()[0, :, :, :].transpose((1, 2, 0))

            # Resize 'ab' channels back to original frame size
            ab = cv2.resize(ab, (width, height))

            # Get the original L channel (from the original frame size) and combine with predicted 'ab'
            L_orig = cv2.split(lab)[0]
            colorized_frame = np.concatenate((L_orig[:, :, np.newaxis], ab), axis=2)

            # Convert back to BGR
            colorized_frame = cv2.cvtColor(colorized_frame, cv2.COLOR_LAB2BGR)
            colorized_frame = np.clip(colorized_frame, 0, 1)
            colorized_frame = (255 * colorized_frame).astype("uint8")

            # Write the colorized frame
            out.write(colorized_frame)

            # Update progress
            if progress_callback:
                 progress_callback(i + 1, frame_count)

            # Optional: Display current frame for debugging/visualization (can be slow)
            # cv2.imshow("Colorizing Video", colorized_frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

        # Release everything when job is finished
        cap.release()
        out.release()
        # cv2.destroyAllWindows() # Close any preview window

        print(f"[INFO] Video colorized and saved to {output_path}")
        return True

    except Exception as e:
        print(f"[ERROR] Error during video colorization: {e}")
        # Ensure resources are released even on error
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        if 'out' in locals() and out.isOpened():
            out.release()
        messagebox.showerror("Error", f"An error occurred during video colorization:\n{e}")
        return False


# --- GUI Implementation ---
class ColorizerApp:
    def __init__(self, root):
        self.root = root
        root.title("BW Image/Video Colorizer")

        self.input_path = tk.StringVar()
        self.status_text = tk.StringVar()
        self.status_text.set("Select a file and click Colorize")

        # --- GUI Layout ---
        main_frame = ttk.Frame(root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Input File Selection
        ttk.Label(main_frame, text="Input File:").grid(row=0, column=0, sticky=tk.W, pady=5)
        input_entry = ttk.Entry(main_frame, textvariable=self.input_path, width=50)
        input_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5, pady=5)
        ttk.Button(main_frame, text="Browse", command=self.browse_file).grid(row=0, column=2, sticky=tk.W, pady=5)

        # Action Button
        ttk.Button(main_frame, text="Colorize", command=self.start_colorization).grid(row=1, column=0, columnspan=3, pady=10)

        # Status Label
        ttk.Label(main_frame, textvariable=self.status_text).grid(row=2, column=0, columnspan=3, sticky=tk.W, pady=5)

        # Progress Bar (for video)
        self.progress_bar = ttk.Progressbar(main_frame, orient="horizontal", length=400, mode="determinate")
        self.progress_bar.grid(row=3, column=0, columnspan=3, pady=5, sticky=(tk.W, tk.E))
        self.progress_bar["value"] = 0 # Initialize progress bar


        # Configure grid weights
        main_frame.columnconfigure(1, weight=1)

    def browse_file(self):
        """Opens a file dialog to select an image or video file."""
        file_path = filedialog.askopenfilename(
            title="Select Image or Video File",
            filetypes=(("Media files", "*.jpg *.jpeg *.png *.mp4 *.avi *.mov"),
                       ("Image files", "*.jpg *.jpeg *.png"),
                       ("Video files", "*.mp4 *.avi *.mov"),
                       ("All files", "*.*"))
        )
        if file_path:
            self.input_path.set(file_path)
            self.status_text.set(f"File selected: {os.path.basename(file_path)}")
            self.progress_bar["value"] = 0 # Reset progress bar

    def start_colorization(self):
        """Starts the colorization process based on the selected file type."""
        input_file = self.input_path.get()
        if not input_file:
            messagebox.showwarning("Warning", "Please select an image or video file first.")
            return

        if not os.path.exists(input_file):
             messagebox.showerror("Error", f"Input file not found: {input_file}")
             return

        # Determine file type
        file_extension = os.path.splitext(input_file)[1].lower()
        is_video = file_extension in ['.mp4', '.avi', '.mov', '.mkv'] # Added .mkv

        if is_video:
            output_file = filedialog.asksaveasfilename(
                title="Save Colorized Video As",
                defaultextension=file_extension,
                filetypes=(("Video files", f"*{file_extension}"), ("All files", "*.*"))
            )
            if output_file:
                self.status_text.set("Colorizing video...")
                self.progress_bar["value"] = 0 # Reset progress bar
                # Running video processing in a separate thread is ideal for responsiveness,
                # but for simplicity, we run it in the main thread.
                # The GUI might become unresponsive during video processing.
                success = colorize_video(input_file, output_file, self.update_progress)
                if success:
                    self.status_text.set(f"Video colorized and saved to {os.path.basename(output_file)}")
                    messagebox.showinfo("Success", f"Video colorized and saved to:\n{output_file}")
                else:
                     self.status_text.set("Video colorization failed.")

        else: # Assume image
            output_file = filedialog.asksaveasfilename(
                title="Save Colorized Image As",
                defaultextension=".png", # Default to png
                filetypes=(("PNG files", "*.png"), ("JPEG files", "*.jpg *.jpeg"), ("All files", "*.*"))
            )
            if output_file:
                self.status_text.set("Colorizing image...")
                # Clear progress bar for images
                self.progress_bar["value"] = 0
                success = colorize_image(input_file, output_file)
                if success:
                    self.status_text.set(f"Image colorized and saved to {os.path.basename(output_file)}")
                    messagebox.showinfo("Success", f"Image colorized and saved to:\n{output_file}")
                else:
                    self.status_text.set("Image colorization failed.")

    def update_progress(self, current_frame, total_frames):
        """Updates the progress bar and status for video colorization."""
        progress_percentage = (current_frame / total_frames) * 100
        self.progress_bar["value"] = progress_percentage
        self.status_text.set(f"Colorizing video: Frame {current_frame}/{total_frames}")
        self.root.update_idletasks() # Update the GUI

if __name__ == "__main__":
    root = tk.Tk()
    app = ColorizerApp(root)
    root.mainloop()