# ImageAI Enhancer

A web application that enhances images and generates captions using machine learning.

## Features

- **Image Enhancement**: Improve image quality using the LapSRN model
- **Image Captioning**: Generate accurate descriptions of images using a Vision Transformer model
- **User Authentication**: Secure login and signup functionality
- **Multiple Image Sources**: Upload images from your device or capture them with your camera
- **Modern UI**: Beautiful and responsive interface with smooth animations and transitions
- **Download Options**: Save enhanced images and copy generated captions

## Technologies Used

- **Backend**: Flask (Python)
- **Frontend**: HTML, CSS, JavaScript
- **Machine Learning**: TensorFlow, PyTorch, Transformers
- **Authentication**: Flask-Login
- **Image Processing**: OpenCV, Pillow

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/image-enhancer.git
   cd image-enhancer
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Download the LapSRN model:
   ```
   mkdir -p models
   wget -O models/LapSRN_x8.pb https://github.com/fannymonori/TF-LapSRN/raw/master/export/LapSRN_x8.pb
   ```

5. Create the uploads directory:
   ```
   mkdir -p static/uploads
   ```

## Usage

1. Start the Flask development server:
   ```
   python app.py
   ```

2. Open your web browser and navigate to:
   ```
   http://127.0.0.1:5000/
   ```

3. Create an account or log in if you already have one.

4. From the dashboard, choose to either upload an image or capture one with your camera.

5. Wait for the image to be processed (enhanced and captioned).

6. View the results and download the enhanced image or copy the generated caption.

## Project Structure

```
image-enhancer/
├── app.py                  # Main Flask application
├── requirements.txt        # Python dependencies
├── models/                 # ML models
│   └── LapSRN_x8.pb        # Image enhancement model
├── static/                 # Static files
│   ├── css/                # CSS stylesheets
│   │   └── style.css       # Main stylesheet
│   ├── js/                 # JavaScript files
│   │   └── main.js         # Main JavaScript file
│   ├── images/             # Static images
│   └── uploads/            # User uploaded images
└── templates/              # HTML templates
    ├── base.html           # Base template
    ├── index.html          # Welcome page
    ├── login.html          # Login page
    ├── signup.html         # Signup page
    ├── dashboard.html      # User dashboard
    ├── image_source.html   # Image source selection
    ├── upload_image.html   # Upload image page
    ├── capture_image.html  # Capture image page
    └── results.html        # Results page
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- [TF-LapSRN](https://github.com/fannymonori/TF-LapSRN) for the image enhancement model
- [ViT-GPT2 Image Captioning](https://huggingface.co/nlpconnect/vit-gpt2-image-captioning) for the image captioning model 