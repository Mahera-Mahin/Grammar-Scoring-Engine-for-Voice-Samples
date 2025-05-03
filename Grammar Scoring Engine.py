#Demonstration :- https://drive.google.com/file/d/1d_5oHVP3XHxugWX3zwZi-Qru_VitvFEB/view?usp=sharing
#Github :- https://github.com/DarshanKagi/Grammar-Scoring-Engine

import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from faster_whisper import WhisperModel  # Changed from whisper to faster_whisper
import pylint.lint                 # For grammar and style checking
import sounddevice as sd           # For audio recording
import soundfile as sf             # For audio file handling
import numpy as np                 # For numerical operations on audio data
import os                          # For file operations
import time                        # For timing and recording duration
import io                          # For capturing pylint output
import sys                         # For redirecting stdout
import torch                       # For GPU detection
import requests                    # For downloading model
from tqdm import tqdm             # For download progress
import subprocess                  # For running system commands
from datetime import datetime
import psutil
import gc
from sklearn.metrics import precision_score, recall_score, f1_score, mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr

# ---------------------------
# Global Constants & Helpers
# ---------------------------

# Error weights used to calculate a weighted penalty for different error categories.
ERROR_WEIGHTS = {
    'GRAMMAR': 0.3,  # Reduced from 3 to 0.3 for 10-point scale
    'TYPOS': 0.2,    # Reduced from 2 to 0.2 for 10-point scale
    'PUNCTUATION': 0.1,  # Reduced from 1 to 0.1 for 10-point scale
    'STYLE': 0.15    # Reduced from 1.5 to 0.15 for 10-point scale
}

# List of filler words to count in the transcript.
FILLER_WORDS = ["um", "uh", "like", "you know", "ah", "er"]

# Dictionary to store inline error matches using tag names in the transcript text widget.
inline_errors = {}

def classify_error(error_type):
    """
    Classify a grammar/style error into one of the categories:
    'GRAMMAR', 'TYPOS', 'PUNCTUATION', or 'STYLE'
    
    Args:
        error_type: A string representing the pylint error type.
    
    Returns:
        A string representing the error category.
    """
    if 'spelling' in error_type.lower():
        return "TYPOS"
    elif 'punctuation' in error_type.lower():
        return "PUNCTUATION"
    elif 'style' in error_type.lower():
        return "STYLE"
    else:
        return "GRAMMAR"

def check_grammar(text):
    """
    Check the provided text for grammar and style issues using pylint.
    
    Args:
        text: The transcript text to check.
    
    Returns:
        A tuple containing:
         - overall_score: The final score after applying weighted penalties (out of 10).
         - error_details: A list of formatted error details.
         - error_breakdown: A dictionary with error counts by category.
         - total_errors: The total number of errors detected.
    """
    # Create a temporary file to store the text
    temp_file = "temp_transcript.py"
    with open(temp_file, "w", encoding="utf-8") as f:
        # Wrap the text in a function to make it valid Python code
        f.write("def transcript():\n")
        for line in text.split('\n'):
            f.write(f"    {line}\n")
    
    # Capture pylint output
    old_stdout = sys.stdout
    new_stdout = io.StringIO()
    sys.stdout = new_stdout
    
    # Run pylint with specific checks
    pylint.lint.Run([temp_file, '--disable=all', '--enable=C,E,W,R', '--exit-zero'], exit=False)
    
    # Restore stdout
    sys.stdout = old_stdout
    output = new_stdout.getvalue()
    
    # Clean up temporary file
    os.remove(temp_file)
    
    error_details = []
    error_breakdown = {"GRAMMAR": 0, "TYPOS": 0, "PUNCTUATION": 0, "STYLE": 0}
    
    # Process pylint output
    for line in output.split('\n'):
        if ':' in line and ':' in line.split(':')[1:]:
            parts = line.split(':')
            if len(parts) >= 3:
                error_type = parts[2].strip()
                category = classify_error(error_type)
                error_breakdown[category] += 1
                error_info = (
                    f"Error: {error_type}\n"
                    f"Category: {category}\n"
                    f"Line: {parts[1]}\n"
                    "-----"
                )
                error_details.append(error_info)
    
    # Calculate weighted penalty based on error counts and predefined weights
    weighted_penalty = sum(ERROR_WEIGHTS[cat] * count for cat, count in error_breakdown.items())
    overall_score = max(10 - weighted_penalty, 0)  # Changed from 100 to 10
    return overall_score, error_details, error_breakdown, sum(error_breakdown.values())

def download_model(url, filename):
    """
    Download a file with progress bar
    """
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)
    
    with open(filename, 'wb') as f:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            f.write(data)
    progress_bar.close()

def check_nvidia_smi():
    """
    Check NVIDIA GPU using nvidia-smi command
    """
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("\nNVIDIA-SMI Output:")
            print(result.stdout)
            return True
        return False
    except Exception as e:
        print(f"Error running nvidia-smi: {e}")
        return False

def get_gpu_info():
    """
    Get detailed GPU information
    """
    print("\nGPU Detection Details:")
    print("-" * 50)
    
    # Check CUDA availability
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        
        # Print information about each GPU
        for i in range(torch.cuda.device_count()):
            print(f"\nGPU {i}:")
            print(f"  Name: {torch.cuda.get_device_name(i)}")
            print(f"  Capability: {torch.cuda.get_device_capability(i)}")
            print(f"  Memory allocated: {torch.cuda.memory_allocated(i) / 1024**2:.2f} MB")
            print(f"  Memory cached: {torch.cuda.memory_reserved(i) / 1024**2:.2f} MB")
    else:
        print("CUDA is not available. Checking NVIDIA-SMI...")
        check_nvidia_smi()
    
    print("-" * 50)

def get_optimal_device():
    """
    Determine the best available device for processing.
    Returns a tuple of (device, compute_type) for optimal performance.
    """
    # Print detailed GPU information
    get_gpu_info()
    
    if torch.cuda.is_available():
        # Get GPU information
        gpu_name = torch.cuda.get_device_name(0)
        print(f"\nUsing GPU: {gpu_name}")
        
        # Set CUDA device properties for optimal performance
        torch.cuda.set_device(0)  # Use first GPU
        torch.backends.cudnn.benchmark = True  # Enable cuDNN auto-tuner
        torch.backends.cudnn.enabled = True    # Enable cuDNN
        
        # For RTX 2050, use float16 for best performance
        return "cuda", "float16"
    elif torch.backends.mps.is_available():
        print("Apple MPS detected: Using MPS for processing")
        return "mps", "float16"
    else:
        print("\nNo GPU detected. Please check:")
        print("1. NVIDIA drivers are installed")
        print("2. CUDA toolkit is installed")
        print("3. PyTorch is installed with CUDA support")
        print("\nFalling back to CPU processing...")
        return "cpu", "int8"

# ---------------------------
# Initialize Models & Tools
# ---------------------------

# Load the Whisper model for audio transcription with optimal device settings
print("Loading Whisper model...")
device, compute_type = get_optimal_device()

# Create models directory if it doesn't exist
models_dir = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(models_dir, exist_ok=True)

try:
    # Try to load the model from local cache first
    model = WhisperModel(
        "tiny",
        device=device,
        compute_type=compute_type,
        download_root=models_dir,
        local_files_only=True,  # Try local files first
        num_workers=4,          # Optimize for RTX 2050
        cpu_threads=4           # Use multiple CPU threads for preprocessing
    )
except Exception as e:
    print(f"Local model not found, downloading from Hugging Face...")
    try:
        # Download model files directly
        model_url = "https://huggingface.co/guillaumekln/faster-whisper-tiny/resolve/main/model.bin"
        model_path = os.path.join(models_dir, "model.bin")
        
        if not os.path.exists(model_path):
            print("Downloading model file...")
            download_model(model_url, model_path)
        
        # Now try loading the model again with GPU optimizations
        model = WhisperModel(
            "tiny",
            device=device,
            compute_type=compute_type,
            download_root=models_dir,
            num_workers=4,      # Optimize for RTX 2050
            cpu_threads=4       # Use multiple CPU threads for preprocessing
        )
    except Exception as e:
        error_msg = f"Failed to load model: {str(e)}\nPlease check your internet connection and try again."
        print(error_msg)
        messagebox.showerror("Model Loading Error", error_msg)
        sys.exit(1)

# ---------------------------
# Audio Recording Setup
# ---------------------------

# Set the sample rate for audio recording (in Hertz)
SAMPLE_RATE = 44100
# Temporary file name for saving recorded audio
TEMP_AUDIO_FILE = "temp_recording.wav"
# Variables for managing audio recording state
recording_stream = None
recorded_frames = []
recording_start_time = None
timer_job = None

def audio_callback(indata, frames, time_info, status):
    """
    Callback function for the audio input stream.
    
    Args:
        indata: Recorded audio data.
        frames: Number of frames.
        time_info: Dictionary containing timing information.
        status: Status of the recording.
    """
    if status:
        print("Recording Status:", status)
    # Append a copy of the current audio chunk to recorded_frames
    recorded_frames.append(indata.copy())

class PerformanceMetrics:
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.transcription_time = 0
        self.audio_duration = 0
        self.memory_usage = []
        self.gpu_memory_usage = []
        self.segment_times = []
        self.confidence_scores = []
        
    def start(self):
        self.start_time = time.time()
        self.memory_usage = []
        self.gpu_memory_usage = []
        self.segment_times = []
        self.confidence_scores = []
        
    def end(self):
        self.end_time = time.time()
        self.transcription_time = self.end_time - self.start_time
        
    def update_memory_usage(self):
        process = psutil.Process()
        self.memory_usage.append(process.memory_info().rss / 1024 / 1024)  # MB
        if torch.cuda.is_available():
            self.gpu_memory_usage.append(torch.cuda.memory_allocated() / 1024 / 1024)  # MB
            
    def add_segment(self, segment_time, confidence):
        self.segment_times.append(segment_time)
        self.confidence_scores.append(confidence)
        
    def get_metrics(self):
        metrics = {
            "Total Processing Time": f"{self.transcription_time:.2f} seconds",
            "Audio Duration": f"{self.audio_duration:.2f} seconds",
            "Processing Speed": f"{self.audio_duration/self.transcription_time:.2f}x real-time",
            "Average Confidence": f"{np.mean(self.confidence_scores):.2%}",
            "Peak Memory Usage": f"{max(self.memory_usage):.2f} MB",
            "Average Segment Time": f"{np.mean(self.segment_times):.2f} seconds"
        }
        if torch.cuda.is_available():
            metrics["Peak GPU Memory"] = f"{max(self.gpu_memory_usage):.2f} MB"
        return metrics

class ModelMetrics:
    def __init__(self):
        self.predictions = []
        self.actual_values = []
        self.confidence_scores = []
        self.segment_times = []
        self.loss_values = []
        self.true_positives = 0
        self.false_positives = 0
        self.true_negatives = 0
        self.false_negatives = 0
        
    def update_metrics(self, prediction, actual, confidence, loss=None):
        if prediction is not None and actual is not None:
            # Convert to binary classification (1 if confidence > 0.5)
            pred_binary = 1 if float(prediction) > 0.5 else 0
            actual_binary = 1 if float(actual) > 0.5 else 0
            
            # Update confusion matrix
            if pred_binary == 1 and actual_binary == 1:
                self.true_positives += 1
            elif pred_binary == 1 and actual_binary == 0:
                self.false_positives += 1
            elif pred_binary == 0 and actual_binary == 0:
                self.true_negatives += 1
            elif pred_binary == 0 and actual_binary == 1:
                self.false_negatives += 1
            
            # Store raw values for other metrics
            self.predictions.append(float(prediction))
            self.actual_values.append(float(actual))
            self.confidence_scores.append(float(confidence))
            if loss is not None:
                self.loss_values.append(float(loss))
            
    def calculate_metrics(self):
        metrics = {}
        try:
            if len(self.predictions) < 2 or len(self.actual_values) < 2:
                return {
                    "Status": "Insufficient data for metrics calculation",
                    "Number of Samples": str(len(self.predictions)),
                    "Average Confidence": f"{np.mean(self.confidence_scores):.4f}" if self.confidence_scores else "N/A"
                }
            
            # Calculate classification metrics
            total = self.true_positives + self.false_positives + self.true_negatives + self.false_negatives
            if total > 0:
                # Calculate accuracy
                accuracy = (self.true_positives + self.true_negatives) / total
                metrics["Accuracy"] = f"{accuracy:.4f}"
                
                # Calculate precision
                precision = self.true_positives / (self.true_positives + self.false_positives) if (self.true_positives + self.false_positives) > 0 else 0
                metrics["Precision"] = f"{precision:.4f}"
                
                # Calculate recall
                recall = self.true_positives / (self.true_positives + self.false_negatives) if (self.true_positives + self.false_negatives) > 0 else 0
                metrics["Recall"] = f"{recall:.4f}"
                
                # Calculate F1 score
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                metrics["F1 Score"] = f"{f1:.4f}"
            
            # Convert to numpy arrays for other calculations
            preds = np.array(self.predictions)
            actuals = np.array(self.actual_values)
            confidences = np.array(self.confidence_scores)
            
            # Basic statistics
            metrics["Number of Samples"] = str(len(preds))
            metrics["Average Confidence"] = f"{np.mean(confidences):.4f}"
            metrics["Confidence Std Dev"] = f"{np.std(confidences):.4f}"
            
            # Error metrics
            metrics["Mean Absolute Error"] = f"{mean_absolute_error(actuals, preds):.4f}"
            metrics["Root Mean Square Error"] = f"{np.sqrt(mean_squared_error(actuals, preds)):.4f}"
            
            # Correlation if we have variation
            if np.std(preds) > 0 and np.std(actuals) > 0:
                correlation, p_value = pearsonr(actuals, preds)
                metrics["Pearson Correlation"] = f"{correlation:.4f}"
                metrics["Correlation P-value"] = f"{p_value:.4f}"
            
            # Loss metrics
            if self.loss_values:
                metrics["Average Loss"] = f"{np.mean(self.loss_values):.4f}"
                metrics["Loss Std Dev"] = f"{np.std(self.loss_values):.4f}"
            
            # Add confusion matrix metrics
            metrics["True Positives"] = str(self.true_positives)
            metrics["False Positives"] = str(self.false_positives)
            metrics["True Negatives"] = str(self.true_negatives)
            metrics["False Negatives"] = str(self.false_negatives)
            
        except Exception as e:
            print(f"Error calculating metrics: {str(e)}")
            metrics["Error"] = f"Error calculating metrics: {str(e)}"
            
        return metrics

def show_metrics_window(metrics):
    """
    Display metrics in a new window
    """
    metrics_window = tk.Toplevel(root)
    metrics_window.title("Performance Metrics")
    metrics_window.geometry("400x500")
    
    # Create a frame with modern styling
    frame = tk.Frame(metrics_window, bg='#f0f0f0', padx=20, pady=20)
    frame.pack(fill="both", expand=True)
    
    # Title
    title = tk.Label(frame, text="Performance Metrics", 
                    font=('Segoe UI', 14, 'bold'),
                    bg='#f0f0f0', fg='#333333')
    title.pack(pady=(0, 20))
    
    # Display each metric
    for key, value in metrics.items():
        metric_frame = tk.Frame(frame, bg='#f0f0f0')
        metric_frame.pack(fill="x", pady=5)
        
        label = tk.Label(metric_frame, text=key + ":", 
                        font=('Segoe UI', 10),
                        bg='#f0f0f0', fg='#666666',
                        anchor="w")
        label.pack(side="left")
        
        value_label = tk.Label(metric_frame, text=value,
                             font=('Segoe UI', 10, 'bold'),
                             bg='#f0f0f0', fg='#333333',
                             anchor="e")
        value_label.pack(side="right")
    
    # Add close button
    close_button = tk.Button(frame, text="Close",
                           command=metrics_window.destroy,
                           bg='#4CAF50', fg='white',
                           font=('Segoe UI', 10, 'bold'),
                           padx=20, pady=10)
    close_button.pack(pady=20)

def show_detailed_metrics_window(metrics):
    """
    Display detailed model metrics in a new window with improved formatting
    """
    metrics_window = tk.Toplevel(root)
    metrics_window.title("Model Performance Metrics")
    metrics_window.geometry("600x800")  # Increased height for more metrics
    
    # Create main frame with padding
    main_frame = tk.Frame(metrics_window, bg='#f0f0f0', padx=20, pady=20)
    main_frame.pack(fill="both", expand=True)
    
    # Title with larger font
    title = tk.Label(main_frame, text="Model Performance Metrics", 
                    font=('Segoe UI', 16, 'bold'),
                    bg='#f0f0f0', fg='#333333')
    title.pack(pady=(0, 20))
    
    # Create canvas and scrollbar for scrolling
    canvas = tk.Canvas(main_frame, bg='#f0f0f0', highlightthickness=0)
    scrollbar = tk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
    scrollable_frame = tk.Frame(canvas, bg='#f0f0f0')
    
    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )
    
    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)
    
    # Define metric categories with their respective metrics
    categories = {
        "Classification Metrics": ["Accuracy", "Precision", "Recall", "F1 Score"],
        "Confusion Matrix": ["True Positives", "False Positives", "True Negatives", "False Negatives"],
        "Basic Statistics": ["Number of Samples", "Average Confidence", "Confidence Std Dev"],
        "Error Metrics": ["Mean Absolute Error", "Root Mean Square Error"],
        "Statistical Metrics": ["Pearson Correlation", "Correlation P-value"],
        "Model Performance": ["Average Loss", "Loss Std Dev"]
    }
    
    # Display metrics in categories
    for category, metric_keys in categories.items():
        # Category header with background color
        category_frame = tk.Frame(scrollable_frame, bg='#e0e0e0', padx=10, pady=5)
        category_frame.pack(fill="x", pady=(10, 5))
        
        category_label = tk.Label(category_frame, text=category,
                                font=('Segoe UI', 12, 'bold'),
                                bg='#e0e0e0', fg='#333333')
        category_label.pack(anchor="w")
        
        # Metrics in this category
        for key in metric_keys:
            if key in metrics:
                metric_frame = tk.Frame(scrollable_frame, bg='#f0f0f0')
                metric_frame.pack(fill="x", pady=2)
                
                # Left side: Metric name
                label = tk.Label(metric_frame, text=key + ":",
                               font=('Segoe UI', 10),
                               bg='#f0f0f0', fg='#666666',
                               anchor="w", width=25)
                label.pack(side="left", padx=5)
                
                # Right side: Metric value
                value_label = tk.Label(metric_frame, text=metrics[key],
                                     font=('Segoe UI', 10, 'bold'),
                                     bg='#f0f0f0', fg='#333333',
                                     anchor="e", width=15)
                value_label.pack(side="right", padx=5)
    
    # Pack canvas and scrollbar
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")
    
    # Add close button with modern styling
    close_button = tk.Button(main_frame, text="Close",
                           command=metrics_window.destroy,
                           bg='#4CAF50', fg='white',
                           font=('Segoe UI', 10, 'bold'),
                           padx=20, pady=10)
    close_button.pack(pady=20)

def transcribe_audio(file_path):
    """
    Transcribe an audio file using the Whisper model with detailed metrics.
    """
    try:
        # Initialize metrics
        metrics = PerformanceMetrics()
        model_metrics = ModelMetrics()
        metrics.start()
        
        # Show a message to indicate transcription is starting
        messagebox.showinfo("Processing", f"Starting transcription using {device.upper()}... This may take a few minutes.")
        root.update()
        
        # Get audio duration
        info = sf.info(file_path)
        metrics.audio_duration = info.duration
        
        print(f"Loading audio file: {file_path}")
        segments, info = model.transcribe(
            file_path,
            beam_size=5,
            language='en',
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
            word_timestamps=False,
            condition_on_previous_text=True,
            initial_prompt="This is a transcription of spoken English.",
            temperature=0.0,
            compression_ratio_threshold=2.4,
            no_speech_threshold=0.6
        )
        
        # Process segments and collect metrics
        transcript_parts = []
        segment_metrics = []
        
        for i, segment in enumerate(segments):
            segment_start = time.time()
            transcript_parts.append(segment.text)
            
            # Calculate segment metrics
            segment_time = time.time() - segment_start
            confidence = np.exp(segment.avg_logprob)  # Convert log probability to probability
            
            # Update performance metrics
            metrics.add_segment(segment_time, confidence)
            metrics.update_memory_usage()
            
            # Store segment metrics
            segment_metrics.append({
                'text': segment.text,
                'confidence': confidence,
                'time': segment_time,
                'loss': -segment.avg_logprob
            })
            
            # Update model metrics
            model_metrics.update_metrics(
                prediction=confidence,
                actual=1.0 if len(segment.text.strip()) > 0 else 0.0,
                confidence=confidence,
                loss=-segment.avg_logprob
            )
            
        # Combine all segments
        transcript = " ".join(transcript_parts)
        
        # End metrics collection
        metrics.end()
        
        # Calculate and show metrics
        performance_metrics = metrics.get_metrics()
        model_metrics_result = model_metrics.calculate_metrics()
        
        # Add segment-level statistics to model metrics
        if segment_metrics:
            confidences = [m['confidence'] for m in segment_metrics]
            times = [m['time'] for m in segment_metrics]
            model_metrics_result.update({
                "Segment Statistics": {
                    "Average Segment Time": f"{np.mean(times):.4f} seconds",
                    "Segment Time Std Dev": f"{np.std(times):.4f} seconds",
                    "Average Segment Confidence": f"{np.mean(confidences):.4f}",
                    "Confidence Std Dev": f"{np.std(confidences):.4f}"
                }
            })
        
        # Show metrics windows
        show_metrics_window(performance_metrics)
        show_detailed_metrics_window(model_metrics_result)
        
        print("Transcription completed successfully")
        return transcript
        
    except Exception as e:
        error_msg = f"Error during transcription: {str(e)}\n\nPlease try a different audio file or check if the file is corrupted."
        print(error_msg)
        messagebox.showerror("Transcription Error", error_msg)
        return ""

def process_file():
    """
    Process an audio file selected from disk:
      - Transcribe audio using Whisper.
      - Compute grammar errors and style metrics.
      - Update GUI text widgets with transcript and score.
    """
    try:
        # Open file dialog to select an audio file
        file_path = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=[("Audio Files", "*.wav *.mp3 *.m4a")]
        )
        
        if file_path:
            print(f"Selected file: {file_path}")
            
            # Update GUI to show processing status
            transcript_text.delete(1.0, tk.END)
            transcript_text.insert(tk.END, "Processing audio file... Please wait...")
            score_label.config(text="Processing...")
            root.update()
            
            # Transcribe the audio file
            transcript = transcribe_audio(file_path)
            
            if transcript:
                try:
                    # Get audio duration
                    info = sf.info(file_path)
                    audio_duration = info.duration
                    print(f"Audio duration: {audio_duration:.2f} seconds")
                except Exception as e:
                    print(f"Could not get audio duration: {e}")
                    audio_duration = 0
                
                # Check grammar on the transcript
                print("Analyzing grammar...")
                overall_score, error_details, error_breakdown, total_errors = check_grammar(transcript)
                
                # Update the transcript text widget with the transcription result
                transcript_text.delete(1.0, tk.END)
                transcript_text.insert(tk.END, transcript)
                
                # Update the score label with overall grammar score and error count
                score_label.config(text=f"Grammar Score: {overall_score:.1f}/10 (Total Errors: {total_errors})")
                print("Processing completed successfully")
            else:
                transcript_text.delete(1.0, tk.END)
                transcript_text.insert(tk.END, "Transcription failed. Please try again with a different audio file.")
                score_label.config(text="Grammar Score: N/A")
    
    except Exception as e:
        error_msg = f"Error processing file: {str(e)}"
        print(error_msg)
        messagebox.showerror("Processing Error", error_msg)
        transcript_text.delete(1.0, tk.END)
        transcript_text.insert(tk.END, "An error occurred while processing the file.")
        score_label.config(text="Grammar Score: N/A")

def update_timer():
    """
    Update the recording timer label every second to show elapsed recording time.
    """
    global timer_job
    if recording_start_time is not None:
        elapsed = int(time.time() - recording_start_time)
        timer_label.config(text=f"Recording Time: {elapsed} sec")
        timer_job = root.after(1000, update_timer)

def start_recording():
    """
    Start audio recording using sounddevice and update the timer.
    """
    global recording_stream, recorded_frames, recording_start_time, timer_job
    recorded_frames = []  # Reset the recorded frames
    recording_start_time = time.time()  # Record the start time
    update_timer()  # Begin updating the timer
    try:
        # Start a new input stream for recording audio
        recording_stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=audio_callback)
        recording_stream.start()
    except Exception as e:
        messagebox.showerror("Recording Error", f"Error starting recording: {e}")

def stop_recording():
    """
    Stop audio recording, process the recorded audio, and generate the analysis report.
    """
    global recording_stream, recorded_frames, recording_start_time, timer_job
    try:
        if recording_stream is not None:
            # Stop and close the recording stream
            recording_stream.stop()
            recording_stream.close()
            recording_stream = None
            if timer_job:
                root.after_cancel(timer_job)
                timer_job = None
            # Calculate the duration of the recorded audio
            audio_duration = time.time() - recording_start_time
            if recorded_frames:
                # Concatenate recorded frames and save to a temporary audio file
                audio_data = np.concatenate(recorded_frames, axis=0)
                sf.write(TEMP_AUDIO_FILE, audio_data, SAMPLE_RATE)
                timer_label.config(text="Recording Stopped")
                
                # Transcribe the recorded audio
                transcript = transcribe_audio(TEMP_AUDIO_FILE)
                overall_score, error_details, error_breakdown, total_errors = check_grammar(transcript)
                
                # Update GUI with the results
                transcript_text.delete(1.0, tk.END)
                transcript_text.insert(tk.END, transcript)
                score_label.config(text=f"Grammar Score: {overall_score:.1f}/10 (Total Errors: {total_errors})")
                
                # Clean up temporary audio file
                try:
                    os.remove(TEMP_AUDIO_FILE)
                except Exception:
                    pass
            else:
                timer_label.config(text="No audio data recorded.")
        else:
            timer_label.config(text="Recording was not started.")
    except Exception as e:
        messagebox.showerror("Recording Error", f"Error stopping recording: {e}")

def clear_inline_errors():
    """
    Remove all inline error tags from the transcript text widget and clear the inline_errors dictionary.
    """
    for tag in list(inline_errors.keys()):
        transcript_text.tag_delete(tag)
    inline_errors.clear()

def error_click_callback(event, tag):
    """
    Callback function triggered when a highlighted error is clicked.
    Opens a popup with error details and suggested corrections.
    
    Args:
        event: The Tkinter event object.
        tag: The tag identifier for the error.
    """
    match = inline_errors.get(tag)
    if not match:
        return

    # Create a popup window for displaying error details and suggestions
    popup = tk.Toplevel(root)
    popup.title("Error Correction")
    popup.geometry("400x250")
    
    # Display error message and context in the popup
    error_info = f"Error: {match}\nContext: '{match}'"
    tk.Label(popup, text=error_info, wraplength=380, justify="left").pack(pady=10)
    
    suggestion_var = tk.StringVar(value="")
    
    # If suggestions are available, display them as radio buttons for selection
    suggestions = [match]  # Assuming a single suggestion for simplicity
    tk.Label(popup, text="Select a suggestion:").pack()
    for suggestion in suggestions:
        tk.Radiobutton(popup, text=suggestion, variable=suggestion_var, value=suggestion).pack(anchor="w")
    
    def apply_correction():
        """
        Apply the selected correction by replacing the error text in the transcript.
        """
        suggestion = suggestion_var.get()
        if suggestion:
            # Get current indices of the error tag in the transcript text widget
            ranges = transcript_text.tag_ranges(tag)
            if ranges:
                start_index = ranges[0]
                end_index = ranges[1]
                # Replace the error text with the chosen suggestion
                transcript_text.delete(start_index, end_index)
                transcript_text.insert(start_index, suggestion)
                # Remove the tag after correction
                transcript_text.tag_delete(tag)
                if tag in inline_errors:
                    del inline_errors[tag]
            popup.destroy()
        else:
            messagebox.showinfo("No Selection", "Please select a suggestion or cancel.")
    
    tk.Button(popup, text="Apply Correction", command=apply_correction).pack(pady=10)
    tk.Button(popup, text="Cancel", command=popup.destroy).pack()

def inline_error_correction():
    """
    Highlight errors directly in the transcript text widget.
    Each highlighted error is clickable, allowing the user to view details and choose a correction.
    """
    clear_inline_errors()  # Clear any existing highlights
    transcript = transcript_text.get("1.0", "end-1c")
    matches = check_grammar(transcript)[1]
    for i, match in enumerate(matches):
        # Calculate the text indices where the error occurs using offset and error length
        start_index = f"1.0+{match.find('Error:') + 6}c"
        end_index = f"1.0+{match.find('Error:') + 6 + match.find('Error:') - 1}c"
        tag = f"error_{i}"
        transcript_text.tag_add(tag, start_index, end_index)
        # Configure the tag to have a light yellow background and underline for visibility
        transcript_text.tag_config(tag, background="lightyellow", underline=1)
        # Bind a left-click event on the tag to trigger the error correction popup
        transcript_text.tag_bind(tag, "<Button-1>", lambda event, tag=tag: error_click_callback(event, tag))
        inline_errors[tag] = match

def show_comparison_mode():
    """
    Open a new window that shows a side-by-side comparison of the original transcript
    and the corrected version. This helps the user see improvements.
    """
    original = transcript_text.get("1.0", "end-1c")
    if not original:
        messagebox.showinfo("Comparison", "No transcript available to compare.")
        return
        
    # Get grammar check results
    grammar_results = check_grammar(original)
    corrected = original  # Default to original if no corrections needed
    
    if grammar_results[1]:  # If there are error details
        corrected = grammar_results[1][0]
    
    # Create a new top-level window for comparison
    comp_window = tk.Toplevel(root)
    comp_window.title("Transcript Comparison")
    comp_window.geometry("900x500")
    
    # Left frame for the original transcript
    left_frame = tk.Frame(comp_window)
    left_frame.pack(side="left", fill="both", expand=True, padx=10, pady=10)
    # Right frame for the corrected transcript
    right_frame = tk.Frame(comp_window)
    right_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)
    
    tk.Label(left_frame, text="Original Transcript").pack()
    orig_text = tk.Text(left_frame, wrap="word", width=45, height=25)
    orig_text.pack(fill="both", expand=True)
    orig_text.insert(tk.END, original)
    
    tk.Label(right_frame, text="Corrected Transcript").pack()
    corr_text = tk.Text(right_frame, wrap="word", width=45, height=25)
    corr_text.pack(fill="both", expand=True)
    corr_text.insert(tk.END, corrected)

def correct_transcript():
    """
    Perform a full correction of the transcript by replacing its entire contents
    with the corrected version from pylint.
    """
    transcript = transcript_text.get("1.0", tk.END).strip()
    if not transcript:
        messagebox.showinfo("Correction", "No transcript available to correct.")
        return
    try:
        grammar_results = check_grammar(transcript)
        corrected = transcript  # Default to original if no corrections needed
        
        if grammar_results[1]:  # If there are error details
            corrected = grammar_results[1][0]
            
        transcript_text.delete("1.0", tk.END)
        transcript_text.insert(tk.END, corrected)
        messagebox.showinfo("Correction", "Transcript has been processed.")
        clear_inline_errors()
    except Exception as e:
        messagebox.showerror("Correction Error", f"Error during correction: {e}")

# ---------------------------
# GUI Construction with Scrollable Canvas
# ---------------------------

try:
    # Initialize the main Tkinter window
    root = tk.Tk()
    root.title("Enhanced Grammar & Style Analysis Engine")
    root.geometry("1200x800")  # Set a larger default window size
    
    # Make sure the window is brought to front
    root.lift()
    root.attributes('-topmost', True)
    root.after_idle(root.attributes, '-topmost', False)
    
    print("GUI initialized successfully")
    
    # Configure styles
    root.configure(bg='#f0f0f0')  # Light gray background
    style = {
        'bg': '#f0f0f0',
        'fg': '#333333',
        'font': ('Segoe UI', 10),
        'padx': 10,
        'pady': 5
    }

    # Create a canvas and a vertical scrollbar to support scrolling for the entire GUI
    main_canvas = tk.Canvas(root, borderwidth=0, bg=style['bg'])
    scrollbar = tk.Scrollbar(root, orient="vertical", command=main_canvas.yview)
    scrollable_frame = tk.Frame(main_canvas, bg=style['bg'])
    
    print("Basic GUI elements created")

    # Update the scroll region when the frame's size changes
    scrollable_frame.bind(
        "<Configure>",
        lambda e: main_canvas.configure(
            scrollregion=main_canvas.bbox("all")
        )
    )

    # Place the scrollable frame inside the canvas
    main_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    main_canvas.configure(yscrollcommand=scrollbar.set)

    main_canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")
    
    print("Scrollable area configured")

    def _on_mousewheel(event):
        """
        Bind the mouse wheel event to scroll the canvas.
        """
        main_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
    main_canvas.bind_all("<MouseWheel>", _on_mousewheel)

    # ---------------------------
    # GUI Control Buttons and Text Areas
    # ---------------------------

    # Create a frame for control buttons with a modern look
    control_frame = tk.Frame(scrollable_frame, bg=style['bg'])
    control_frame.pack(pady=15, fill="x")
    
    print("Control frame created")
    
    # Button style
    button_style = {
        'bg': '#4CAF50',  # Green color
        'fg': 'white',
        'font': ('Segoe UI', 10, 'bold'),
        'padx': 15,
        'pady': 8,
        'borderwidth': 0,
        'relief': 'flat'
    }
    
    # Create buttons with modern styling
    load_button = tk.Button(control_frame, text="Select & Process Audio File", command=process_file, **button_style)
    load_button.grid(row=0, column=0, padx=5, pady=5)
    
    start_rec_button = tk.Button(control_frame, text="Start Recording", command=start_recording, **button_style)
    start_rec_button.grid(row=0, column=1, padx=5, pady=5)
    
    stop_rec_button = tk.Button(control_frame, text="Stop Recording", command=stop_recording, **button_style)
    stop_rec_button.grid(row=0, column=2, padx=5, pady=5)
    
    correct_button = tk.Button(control_frame, text="Full Correction", command=correct_transcript, **button_style)
    correct_button.grid(row=0, column=3, padx=5, pady=5)
    
    inline_button = tk.Button(control_frame, text="Inline Correction", command=inline_error_correction, **button_style)
    inline_button.grid(row=0, column=4, padx=5, pady=5)
    
    compare_button = tk.Button(control_frame, text="Comparison Mode", command=show_comparison_mode, **button_style)
    compare_button.grid(row=0, column=5, padx=5, pady=5)
    
    print("Buttons created")
    
    # Timer label with modern styling
    timer_label = tk.Label(scrollable_frame, text="Recording Time: 0 sec", 
                          font=('Segoe UI', 12, 'bold'), bg=style['bg'], fg='#333333')
    timer_label.pack(pady=10)

    def create_scrollable_text(master, height, width, title):
        """
            Utility function to create a scrollable text widget with a modern look.
        
        Args:
            master: The parent widget.
            height: The height of the text widget.
            width: The width of the text widget.
                title: The title for the section.
        
        Returns:
            The created text widget.
        """
        frame = tk.Frame(master, bg=style['bg'])
        frame.pack(pady=5, fill="x")
        
        # Add a title label
        title_label = tk.Label(frame, text=title, font=('Segoe UI', 12, 'bold'), 
                              bg=style['bg'], fg='#333333')
        title_label.pack(anchor="w", padx=5)
        
        # Create a container for the text widget and scrollbar
        container = tk.Frame(frame, bg='white', bd=2, relief='groove')
        container.pack(fill="both", expand=True, padx=5)
        
        text_widget = tk.Text(container, wrap="word", height=height, width=width,
                             font=('Segoe UI', 10), bg='white', fg='#333333',
                             padx=10, pady=10)
        widget_scrollbar = tk.Scrollbar(container, orient="vertical", 
                                      command=text_widget.yview)
        text_widget.configure(yscrollcommand=widget_scrollbar.set)
        
        text_widget.pack(side="left", fill="both", expand=True)
        widget_scrollbar.pack(side="right", fill="y")
        
        return text_widget

    # ---------------------------
    # GUI Sections for Output Display
    # ---------------------------

    # Transcript Section
    transcript_text = create_scrollable_text(scrollable_frame, height=15, width=90, 
                                           title="Transcript:")
    
    print("Transcript section created")
    
    # Grammar Score Label with modern styling
    score_frame = tk.Frame(scrollable_frame, bg=style['bg'])
    score_frame.pack(pady=10, fill="x")
    score_label = tk.Label(score_frame, text="Grammar Score: N/A", 
                          font=('Segoe UI', 14, 'bold'), bg=style['bg'], fg='#333333')
    score_label.pack()
    
    print("Score label created")

    # Start the Tkinter main event loop
    print("Starting main event loop")
    root.mainloop()
    
except Exception as e:
    print(f"Error initializing GUI: {str(e)}")
    import traceback
    traceback.print_exc()
    input("Press Enter to exit...")
