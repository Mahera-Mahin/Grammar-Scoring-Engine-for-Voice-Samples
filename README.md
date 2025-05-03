# Grammar Scoring Engine

A powerful tool for transcribing audio and analyzing grammar using advanced machine learning models. This application provides real-time transcription, grammar analysis, and detailed performance metrics.

## Features

- **Audio Transcription**: Convert speech to text using the Faster Whisper model
- **Grammar Analysis**: Check grammar and provide detailed feedback
- **Performance Metrics**: View detailed statistics about transcription accuracy and processing
- **GPU Acceleration**: Automatic GPU detection and optimization for faster processing
- **User-Friendly Interface**: Simple and intuitive GUI for easy operation

## Requirements

- Python 3.10 or higher
- NVIDIA GPU (optional, for faster processing)
- Windows 10 or higher

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Grammar-Scoring-Engine.git
cd Grammar-Scoring-Engine
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
.\.venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the application:
```bash
python "Grammar Scoring Engine.py"
```

2. Using the Application:
   - Click "Select Audio File" to choose an audio file for transcription
   - Click "Record Audio" to record new audio directly
   - View the transcription and grammar analysis results
   - Check performance metrics for detailed statistics

## Performance Metrics

The application provides comprehensive metrics including:

### Classification Metrics
- Precision
- Recall
- F1 Score
- Accuracy

### Error Metrics
- Mean Absolute Error (MAE)
- Root Mean Square Error (RMSE)
- Loss Rate

### Statistical Metrics
- Pearson Correlation
- Confidence Scores
- Processing Time

## GPU Support

The application automatically detects and utilizes available GPUs:
- NVIDIA GPUs: 5-10x faster processing
- Apple Silicon: 3-5x faster processing
- CPU fallback with optimized settings

## Troubleshooting

1. If GPU is not detected:
   - Ensure NVIDIA drivers are installed
   - Install CUDA Toolkit
   - Verify PyTorch installation with CUDA support

2. If model download fails:
   - Check internet connection
   - Verify sufficient disk space
   - Try running as administrator

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- Faster Whisper for the transcription model
- Language Tool for grammar analysis
- PyTorch for GPU acceleration
