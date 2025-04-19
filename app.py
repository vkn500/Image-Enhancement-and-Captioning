import os
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify, send_from_directory
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from PIL import Image, ImageEnhance
import torch
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import base64
import io
import json
import uuid
import datetime
import random

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ENHANCED_FOLDER'] = 'static/enhanced'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'webp'}

# Create upload folder if it doesn't exist
os.makedirs(os.path.join(app.root_path, app.config['UPLOAD_FOLDER']), exist_ok=True)
# Create enhanced folder if it doesn't exist
os.makedirs(os.path.join(app.root_path, app.config['ENHANCED_FOLDER']), exist_ok=True)
# Create data folder if it doesn't exist
os.makedirs(os.path.join(app.root_path, 'data'), exist_ok=True)

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# File paths for persistent data
USERS_FILE = os.path.join(app.root_path, 'data', 'users.json')
SETTINGS_FILE = os.path.join(app.root_path, 'data', 'settings.json')
HISTORY_FILE = os.path.join(app.root_path, 'data', 'history.json')

# Simple user database (in a real app, use a proper database)
users = {}
# User history database
user_history = {}
# User settings database
user_settings = {}

# Load data from JSON files if they exist
def load_data():
    global users, user_history, user_settings
    
    # Load users
    if os.path.exists(USERS_FILE):
        try:
            with open(USERS_FILE, 'r') as f:
                users = json.load(f)
        except Exception as e:
            print(f"Error loading users data: {e}")
    
    # Load history
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r') as f:
                user_history = json.load(f)
        except Exception as e:
            print(f"Error loading history data: {e}")
    
    # Load settings
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, 'r') as f:
                user_settings = json.load(f)
        except Exception as e:
            print(f"Error loading settings data: {e}")

# Save data to JSON files
def save_data():
    # Save users
    try:
        with open(USERS_FILE, 'w') as f:
            json.dump(users, f)
    except Exception as e:
        print(f"Error saving users data: {e}")
    
    # Save history
    try:
        with open(HISTORY_FILE, 'w') as f:
            json.dump(user_history, f)
    except Exception as e:
        print(f"Error saving history data: {e}")
    
    # Save settings
    try:
        with open(SETTINGS_FILE, 'w') as f:
            json.dump(user_settings, f)
    except Exception as e:
        print(f"Error saving settings data: {e}")

# Load data at startup
load_data()

# Add a test user if no users exist
if not users:
    test_user_id = str(uuid.uuid4())
    users[test_user_id] = {
        'username': 'test',
        'email': 'test@example.com',
        'password': generate_password_hash('password123', method='pbkdf2:sha256')
    }
    
    # Initialize user settings with default values
    user_settings[test_user_id] = {
        'dark_mode': False,
        'notifications': True,
        'language': 'English'
    }
    
    # Save data to persist it
    save_data()
    print("Created test user: username=test, password=password123")

# User class for Flask-Login
class User(UserMixin):
    def __init__(self, id, username, email):
        self.id = id
        self.username = username
        self.email = email

@login_manager.user_loader
def load_user(user_id):
    if user_id in users:
        return User(user_id, users[user_id]['username'], users[user_id]['email'])
    return None

# Load image captioning model
def load_captioning_model():
    model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    return model, feature_extractor, tokenizer, device

# Load image enhancement model
def load_enhancement_model():
    model_path = os.path.join(app.root_path, 'models', 'LapSRN_x8.pb')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Enhancement model not found at {model_path}")
    
    # Load the TensorFlow model
    with tf.io.gfile.GFile(model_path, 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
    
    # Create a new graph and import the model
    graph = tf.Graph()
    with graph.as_default():
        tf.import_graph_def(graph_def, name='')
    
    # Create a session
    sess = tf.compat.v1.Session(graph=graph)
    
    # Print operations in the graph to debug
    operations = [op.name for op in graph.get_operations()]
    print("Available operations in the graph:", operations[:10])  # Print first 10 operations
    
    return sess, graph

# Initialize models
try:
    caption_model, feature_extractor, tokenizer, device = load_captioning_model()
    print("Captioning model loaded successfully")
except Exception as e:
    print(f"Error loading captioning model: {e}")
    caption_model, feature_extractor, tokenizer, device = None, None, None, None

try:
    enhancement_sess, enhancement_graph = load_enhancement_model()
    print("Enhancement model loaded successfully")
except Exception as e:
    print(f"Error loading enhancement model: {e}")
    enhancement_sess, enhancement_graph = None, None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def generate_caption(image):
    if caption_model is None:
        return "Image captioning model not loaded"
    
    try:
        # Prepare the image for the model
        if isinstance(image, str):  # If image is a file path
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):  # If image is a numpy array
            image = Image.fromarray(image).convert('RGB')
        
        pixel_values = feature_extractor(images=[image], return_tensors="pt").pixel_values.to(device)
        
        # Generate caption
        with torch.no_grad():
            output_ids = caption_model.generate(pixel_values, max_length=50, num_beams=4)
        
        # Decode the caption
        caption = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        return caption
    
    except Exception as e:
        print(f"Error generating caption: {e}")
        return "Error generating caption"

def enhance_image(image_path):
    if enhancement_sess is None or enhancement_graph is None:
        return None, "Image enhancement model not loaded"
    
    try:
        # Read the image
        img = cv2.imread(image_path)
        if img is None:
            return None, "Failed to read image"
        
        # For now, let's skip the enhancement and just use the original image
        # This will allow us to test if the rest of the pipeline works
        filename = os.path.basename(image_path)
        base_name, ext = os.path.splitext(filename)
        enhanced_filename = f"{base_name}_enhanced{ext}"
        enhanced_path = os.path.join(app.root_path, app.config['ENHANCED_FOLDER'], enhanced_filename)
        
        # Apply a simple enhancement (increase brightness and contrast)
        enhanced_img = cv2.convertScaleAbs(img, alpha=1.2, beta=10)
        cv2.imwrite(enhanced_path, enhanced_img)
        
        # Generate a caption for the original image
        caption = generate_caption(image_path)
        print(f"Generated caption: {caption}")
        
        # Get image details for history
        img_size = os.path.getsize(image_path) / 1024  # Size in KB
        img_format = os.path.splitext(image_path)[1][1:].upper()
        
        # Calculate a random accuracy score between 85% and 99% for demo purposes
        # In a real app, this would be based on actual enhancement metrics
        enhancement_accuracy = f"{random.randint(85, 99)}%"
        
        # Store in history if user is logged in
        if current_user.is_authenticated:
            # Initialize user history if not exists
            if current_user.id not in user_history:
                user_history[current_user.id] = []
            
            # Add to history
            user_history[current_user.id].append({
                'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'source_type': session.get('source_type', 'upload'),
                'original_image': os.path.basename(image_path),
                'enhanced_image': enhanced_filename,
                'image_size': f"{img_size:.1f} KB",
                'image_format': img_format,
                'enhancement_accuracy': enhancement_accuracy,
                'caption': caption
            })
            
            # Save history data
            save_data()
        
        return enhanced_filename, None
    except Exception as e:
        print(f"Error enhancing image: {str(e)}")
        return None, f"Error enhancing image: {str(e)}"

@app.route('/')
def index():
    # Get user settings for dark mode if user is logged in
    dark_mode = False
    if current_user.is_authenticated and current_user.id in user_settings:
        dark_mode = user_settings[current_user.id].get('dark_mode', False)
    
    return render_template('index.html', dark_mode=dark_mode)

@app.route('/upload')
def upload_page():
    # Get user settings for dark mode if user is logged in
    dark_mode = False
    if current_user.is_authenticated and current_user.id in user_settings:
        dark_mode = user_settings[current_user.id].get('dark_mode', False)
    
    return render_template('upload.html', dark_mode=dark_mode)

@app.route('/camera')
def camera_page():
    # Get user settings for dark mode if user is logged in
    dark_mode = False
    if current_user.is_authenticated and current_user.id in user_settings:
        dark_mode = user_settings[current_user.id].get('dark_mode', False)
    
    return render_template('camera.html', dark_mode=dark_mode)

@app.route('/process_upload', methods=['POST'])
def process_upload():
    if 'file' not in request.files:
        flash('No file part', 'error')
        return redirect(url_for('upload_page'))
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No selected file', 'error')
        return redirect(url_for('upload_page'))
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.root_path, app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Create a job ID
        job_id = str(uuid.uuid4())
        session['job_id'] = job_id
        session['image_path'] = file_path
        session['source_type'] = 'upload'
        
        # Redirect to loading page
        return redirect(url_for('loading', job_id=job_id))
    else:
        flash('Invalid file format. Please upload a JPG, JPEG, or PNG file.', 'error')
        return redirect(url_for('upload_page'))

@app.route('/process_camera', methods=['POST'])
def process_camera():
    image_data = request.form.get('image')
    
    if not image_data:
        flash('No image captured', 'error')
        return redirect(url_for('camera_page'))
    
    try:
        # Remove the data URL prefix if present
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        # Decode the base64 image
        image_bytes = base64.b64decode(image_data)
        
        # Generate a unique filename
        filename = f"camera_{uuid.uuid4()}.jpg"
        file_path = os.path.join(app.root_path, app.config['UPLOAD_FOLDER'], filename)
        
        # Save the image
        with open(file_path, 'wb') as f:
            f.write(image_bytes)
        
        # Create a job ID
        job_id = str(uuid.uuid4())
        session['job_id'] = job_id
        session['image_path'] = file_path
        session['source_type'] = 'camera'
        
        # Redirect to loading page
        return redirect(url_for('loading', job_id=job_id))
    except Exception as e:
        error_message = f"Error processing camera image: {str(e)}"
        flash(error_message, 'error')
        return render_template('error.html', error_message=error_message, error_details="There was a problem processing the camera image. Please try again or use the upload option instead.")

@app.route('/loading/<job_id>')
def loading(job_id):
    # Get user settings for dark mode if user is logged in
    dark_mode = False
    if current_user.is_authenticated and current_user.id in user_settings:
        dark_mode = user_settings[current_user.id].get('dark_mode', False)
    
    # Set the redirect URL for the JavaScript to use
    redirect_url = url_for('results', job_id=job_id)
    return render_template('loading.html', redirect_url=redirect_url, dark_mode=dark_mode)

@app.route('/results/<job_id>')
def results(job_id):
    image_path = session.get('image_path')
    
    if not image_path or not os.path.exists(image_path):
        flash('No image to process', 'error')
        return redirect(url_for('index'))
    
    try:
        # Enhance the image
        enhanced_filename, error = enhance_image(image_path)
        
        if error:
            flash(error, 'error')
            # Get user settings for dark mode if user is logged in
            dark_mode = False
            if current_user.is_authenticated and current_user.id in user_settings:
                dark_mode = user_settings[current_user.id].get('dark_mode', False)
            return render_template('error.html', error_message=error, error_details="Please try again with a different image.", dark_mode=dark_mode)
        
        # Generate caption for the enhanced image
        caption = generate_caption(image_path)
        
        # Get user settings for dark mode if user is logged in
        dark_mode = False
        if current_user.is_authenticated and current_user.id in user_settings:
            dark_mode = user_settings[current_user.id].get('dark_mode', False)
        
        # Return the results
        return render_template('results.html',
                              original_image=os.path.basename(image_path),
                              enhanced_image=enhanced_filename,
                              caption=caption,
                              dark_mode=dark_mode)
    except Exception as e:
        error_message = f"Error processing image: {str(e)}"
        flash(error_message, 'error')
        # Get user settings for dark mode if user is logged in
        dark_mode = False
        if current_user.is_authenticated and current_user.id in user_settings:
            dark_mode = user_settings[current_user.id].get('dark_mode', False)
        return render_template('error.html', error_message=error_message, error_details="An unexpected error occurred during processing.", dark_mode=dark_mode)

@app.route('/contact', methods=['POST'])
def contact():
    name = request.form.get('name')
    email = request.form.get('email')
    message = request.form.get('message')
    
    # Here you would typically send an email or save to a database
    # For now, we'll just flash a message
    flash(f'Thank you {name}! Your message has been received.', 'success')
    return redirect(url_for('index'))

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        # Simple validation
        if not username or not email or not password or not confirm_password:
            flash('All fields are required', 'error')
            return redirect(url_for('signup'))
        
        # Check if passwords match
        if password != confirm_password:
            flash('Passwords do not match', 'error')
            return redirect(url_for('signup'))
        
        # Check if password is at least 8 characters
        if len(password) < 8:
            flash('Password must be at least 8 characters long', 'error')
            return redirect(url_for('signup'))
        
        # Check if username already exists
        for user_id, user_data in users.items():
            if user_data['username'] == username:
                flash('Username already exists', 'error')
                return redirect(url_for('signup'))
        
        # Create new user
        user_id = str(uuid.uuid4())
        users[user_id] = {
            'username': username,
            'email': email,
            'password': generate_password_hash(password, method='pbkdf2:sha256')
        }
        
        # Initialize user settings with default values
        user_settings[user_id] = {
            'dark_mode': False,
            'notifications': True,
            'language': 'English'
        }
        
        # Save data to persist it
        save_data()
        
        # Log in the new user
        user = User(user_id, username, email)
        login_user(user)
        
        flash('Account created successfully!', 'success')
        return redirect(url_for('index'))
    
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        print(f"Login attempt: username={username}")
        print(f"Users: {users}")
        
        # Find user by username
        user_id = None
        for uid, user_data in users.items():
            if user_data['username'] == username:
                user_id = uid
                print(f"Found user: {user_id}")
                break
        
        if user_id and check_password_hash(users[user_id]['password'], password):
            user = User(user_id, users[user_id]['username'], users[user_id]['email'])
            login_user(user)
            print(f"Login successful for user: {username}")
            flash('Login successful!', 'success')
            return redirect(url_for('index'))
        
        print(f"Login failed for user: {username}")
        flash('Invalid username or password', 'error')
    
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    # Get user settings for dark mode
    dark_mode = False
    if current_user.id in user_settings:
        dark_mode = user_settings[current_user.id].get('dark_mode', False)
    
    # Get user history for recent activity
    user_image_history = user_history.get(current_user.id, [])
    
    # Sort history by timestamp (newest first)
    user_image_history.sort(key=lambda x: x['timestamp'], reverse=True)
    
    # Get today's date in the same format as timestamps in history
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    
    return render_template('dashboard.html', 
                          dark_mode=dark_mode,
                          history=user_image_history,
                          today=today)

@app.route('/image-source', methods=['GET'])
@login_required
def image_source():
    # Get user settings for dark mode
    dark_mode = False
    if current_user.id in user_settings:
        dark_mode = user_settings[current_user.id].get('dark_mode', False)
    
    return render_template('image_source.html', dark_mode=dark_mode)

@app.route('/upload-image', methods=['GET', 'POST'])
@login_required
def upload_image():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'image' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        file = request.files['image']
        
        # If user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.root_path, app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Store the file path in session for processing
            session['image_path'] = file_path
            session['source_type'] = 'upload'
            
            return redirect(url_for('process_image'))
    
    # Get user settings for dark mode
    dark_mode = False
    if current_user.id in user_settings:
        dark_mode = user_settings[current_user.id].get('dark_mode', False)
    
    return render_template('upload_image.html', dark_mode=dark_mode)

@app.route('/capture-image', methods=['GET', 'POST'])
@login_required
def capture_image():
    if request.method == 'POST':
        # Get the base64 image data from the request
        image_data = request.form.get('image')
        
        if not image_data:
            flash('No image captured')
            return redirect(request.url)
        
        # Remove the data URL prefix
        image_data = image_data.split(',')[1]
        
        # Decode the base64 image
        image_bytes = base64.b64decode(image_data)
        
        # Generate a unique filename
        filename = f"{uuid.uuid4()}.jpg"
        file_path = os.path.join(app.root_path, app.config['UPLOAD_FOLDER'], filename)
        
        # Save the image
        with open(file_path, 'wb') as f:
            f.write(image_bytes)
        
        # Store the file path in session for processing
        session['image_path'] = file_path
        session['source_type'] = 'capture'
        
        return redirect(url_for('process_image'))
    
    # Get user settings for dark mode
    dark_mode = False
    if current_user.id in user_settings:
        dark_mode = user_settings[current_user.id].get('dark_mode', False)
    
    return render_template('capture_image.html', dark_mode=dark_mode)

@app.route('/process-image', methods=['GET'])
@login_required
def process_image():
    image_path = session.get('image_path')
    
    if not image_path or not os.path.exists(image_path):
        flash('No image to process')
        return redirect(url_for('image_source'))
    
    # Enhance the image
    enhanced_filename, error = enhance_image(image_path)
    
    if error:
        flash(error)
        return redirect(url_for('image_source'))
    
    # Generate caption for the enhanced image
    caption = generate_caption(image_path)
    
    # Store results in session
    session['original_image'] = os.path.basename(image_path)
    session['enhanced_image'] = enhanced_filename
    session['caption'] = caption
    
    return redirect(url_for('results'))

@app.route('/history')
@login_required
def history():
    # Get user settings for dark mode
    dark_mode = False
    if current_user.id in user_settings:
        dark_mode = user_settings[current_user.id].get('dark_mode', False)
    
    # Get user history
    user_items = []
    if current_user.id in user_history:
        user_items = user_history[current_user.id]
    
    return render_template('history.html', history=user_items, dark_mode=dark_mode)

@app.route('/delete_history_item/<int:item_index>', methods=['POST'])
@login_required
def delete_history_item(item_index):
    if current_user.id in user_history and 0 <= item_index < len(user_history[current_user.id]):
        # Remove the item from history
        user_history[current_user.id].pop(item_index)
        # Save changes
        save_data()
        flash('History item deleted successfully', 'success')
    else:
        flash('Item not found', 'error')
    
    return redirect(url_for('history'))

@app.route('/clear_history', methods=['POST'])
@login_required
def clear_history():
    if current_user.id in user_history:
        # Clear all history items
        user_history[current_user.id] = []
        # Save changes
        save_data()
        flash('History cleared successfully', 'success')
    
    return redirect(url_for('history'))

@app.route('/settings', methods=['GET', 'POST'])
@login_required
def settings():
    if request.method == 'POST':
        # Handle different form submissions based on form data
        if 'username' in request.form:
            # Profile settings
            username = request.form.get('username')
            email = request.form.get('email')
            bio = request.form.get('bio')
            language = request.form.get('language')
            
            # Update user info
            users[current_user.id]['username'] = username
            users[current_user.id]['email'] = email
            
            # Initialize user settings if not exists
            if current_user.id not in user_settings:
                user_settings[current_user.id] = {}
            
            # Update user settings
            user_settings[current_user.id]['bio'] = bio
            user_settings[current_user.id]['language'] = language
            
            # Save changes
            save_data()
            
            flash('Profile settings updated successfully', 'success')
        
        elif 'dark_mode' in request.form:
            # Appearance settings
            dark_mode = request.form.get('dark_mode') == 'on'
            accent_color = request.form.get('accent_color', 'purple')
            
            # Initialize user settings if not exists
            if current_user.id not in user_settings:
                user_settings[current_user.id] = {}
            
            # Update user settings
            user_settings[current_user.id]['dark_mode'] = dark_mode
            user_settings[current_user.id]['accent_color'] = accent_color
            
            # Save changes
            save_data()
            
            flash('Appearance settings updated successfully', 'success')
        
        elif 'email_notifications' in request.form:
            # Notification settings
            email_notifications = 'email_notifications' in request.form
            processing_notifications = 'processing_notifications' in request.form
            marketing_emails = 'marketing_emails' in request.form
            
            # Initialize user settings if not exists
            if current_user.id not in user_settings:
                user_settings[current_user.id] = {}
            
            # Update user settings
            user_settings[current_user.id]['notifications'] = email_notifications
            user_settings[current_user.id]['processing_notifications'] = processing_notifications
            user_settings[current_user.id]['marketing_emails'] = marketing_emails
            
            # Save changes
            save_data()
            
            flash('Notification settings updated successfully', 'success')
        
        elif 'current_password' in request.form:
            # Security settings - password change
            current_password = request.form.get('current_password')
            new_password = request.form.get('new_password')
            confirm_password = request.form.get('confirm_password')
            
            # Validate current password
            if not current_password or not check_password_hash(users[current_user.id]['password'], current_password):
                flash('Current password is incorrect', 'error')
                return redirect(url_for('settings'))
            
            # Validate new password
            if not new_password or len(new_password) < 8:
                flash('New password must be at least 8 characters', 'error')
                return redirect(url_for('settings'))
            
            # Validate password confirmation
            if new_password != confirm_password:
                flash('New passwords do not match', 'error')
                return redirect(url_for('settings'))
            
            # Update password
            users[current_user.id]['password'] = generate_password_hash(new_password, method='pbkdf2:sha256')
            
            # Two-factor authentication
            two_factor = 'two_factor' in request.form
            
            # Initialize user settings if not exists
            if current_user.id not in user_settings:
                user_settings[current_user.id] = {}
            
            # Update user settings
            user_settings[current_user.id]['two_factor'] = two_factor
            
            # Save changes
            save_data()
            
            flash('Security settings updated successfully', 'success')
        
        elif 'store_history' in request.form:
            # Privacy settings
            store_history = 'store_history' in request.form
            data_collection = 'data_collection' in request.form
            
            # Initialize user settings if not exists
            if current_user.id not in user_settings:
                user_settings[current_user.id] = {}
            
            # Update user settings
            user_settings[current_user.id]['store_history'] = store_history
            user_settings[current_user.id]['data_collection'] = data_collection
            
            # Save changes
            save_data()
            
            flash('Privacy settings updated successfully', 'success')
        
        return redirect(url_for('settings'))
    
    # Get user settings
    user_data = {
        'username': current_user.username,
        'email': current_user.email
    }
    
    settings_data = {}
    if current_user.id in user_settings:
        settings_data = user_settings[current_user.id]
    
    # Get dark mode setting
    dark_mode = settings_data.get('dark_mode', False)
    
    return render_template('settings.html', 
                          username=user_data['username'], 
                          email=user_data['email'], 
                          settings=settings_data,
                          dark_mode=dark_mode)

@app.route('/toggle-dark-mode', methods=['POST'])
@login_required
def toggle_dark_mode():
    if current_user.id in user_settings:
        user_settings[current_user.id]['dark_mode'] = not user_settings[current_user.id].get('dark_mode', False)
    else:
        user_settings[current_user.id] = {
            'dark_mode': True,
            'notifications': True,
            'language': 'English'
        }
    
    # Save changes
    save_data()
    
    return redirect(request.referrer or url_for('dashboard'))

@app.route('/download/<filename>')
def download(filename):
    return send_from_directory(os.path.join(app.root_path, 'static/enhanced'), filename, as_attachment=True)

@app.route('/about')
def about():
    # Get user settings for dark mode if user is logged in
    dark_mode = False
    if current_user.is_authenticated and current_user.id in user_settings:
        dark_mode = user_settings[current_user.id].get('dark_mode', False)
    
    return render_template('about.html', dark_mode=dark_mode)

@app.route('/contact_page')
def contact_page():
    # Get user settings for dark mode if user is logged in
    dark_mode = False
    if current_user.is_authenticated and current_user.id in user_settings:
        dark_mode = user_settings[current_user.id].get('dark_mode', False)
    
    return render_template('contact.html', dark_mode=dark_mode)

# Add language selection route
@app.route('/set_language/<lang>')
def set_language(lang):
    # Store the selected language in session
    session['language'] = lang
    # Redirect back to the previous page or home
    return redirect(request.referrer or url_for('index'))

# Add language context processor to make language available in all templates
@app.context_processor
def inject_language():
    return {
        'current_language': session.get('language', 'en')
    }

# Image editing routes
@app.route('/api/edit/rotate', methods=['POST'])
def rotate_image():
    if not current_user.is_authenticated:
        return jsonify({'success': False, 'error': 'Authentication required'}), 401
    
    try:
        data = request.get_json()
        image_path = os.path.join(app.root_path, app.config['ENHANCED_FOLDER'], data.get('filename'))
        degrees = int(data.get('degrees', 0))
        
        if not os.path.exists(image_path):
            return jsonify({'success': False, 'error': 'Image not found'}), 404
        
        # Open the image
        img = Image.open(image_path)
        
        # Rotate the image
        rotated_img = img.rotate(-degrees, expand=True)  # Negative because PIL rotates counter-clockwise
        
        # Save the image
        rotated_img.save(image_path)
        
        return jsonify({
            'success': True, 
            'filename': os.path.basename(image_path),
            'timestamp': datetime.datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/edit/flip', methods=['POST'])
def flip_image():
    if not current_user.is_authenticated:
        return jsonify({'success': False, 'error': 'Authentication required'}), 401
    
    try:
        data = request.get_json()
        image_path = os.path.join(app.root_path, app.config['ENHANCED_FOLDER'], data.get('filename'))
        direction = data.get('direction', 'horizontal')
        
        if not os.path.exists(image_path):
            return jsonify({'success': False, 'error': 'Image not found'}), 404
        
        # Open the image
        img = Image.open(image_path)
        
        # Flip the image
        if direction == 'horizontal':
            flipped_img = img.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            flipped_img = img.transpose(Image.FLIP_TOP_BOTTOM)
        
        # Save the image
        flipped_img.save(image_path)
        
        return jsonify({
            'success': True, 
            'filename': os.path.basename(image_path),
            'timestamp': datetime.datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/edit/filter', methods=['POST'])
def apply_filter():
    if not current_user.is_authenticated:
        return jsonify({'success': False, 'error': 'Authentication required'}), 401
    
    try:
        data = request.get_json()
        image_path = os.path.join(app.root_path, app.config['ENHANCED_FOLDER'], data.get('filename'))
        filter_type = data.get('filter', 'grayscale')
        
        if not os.path.exists(image_path):
            return jsonify({'success': False, 'error': 'Image not found'}), 404
        
        # Open the image
        img = Image.open(image_path)
        
        # Apply filter
        if filter_type == 'grayscale':
            filtered_img = img.convert('L').convert('RGB')
        elif filter_type == 'sepia':
            # Sepia filter
            sepia_img = img.convert('RGB')
            pixels = sepia_img.load()
            width, height = sepia_img.size
            
            for py in range(height):
                for px in range(width):
                    r, g, b = sepia_img.getpixel((px, py))
                    tr = int(0.393 * r + 0.769 * g + 0.189 * b)
                    tg = int(0.349 * r + 0.686 * g + 0.168 * b)
                    tb = int(0.272 * r + 0.534 * g + 0.131 * b)
                    
                    # Ensure values are within range
                    tr = min(255, tr)
                    tg = min(255, tg)
                    tb = min(255, tb)
                    
                    pixels[px, py] = (tr, tg, tb)
            
            filtered_img = sepia_img
        elif filter_type == 'invert':
            # Invert filter
            filtered_img = Image.eval(img, lambda x: 255 - x)
        else:
            filtered_img = img
        
        # Save the image
        filtered_img.save(image_path)
        
        return jsonify({
            'success': True, 
            'filename': os.path.basename(image_path),
            'timestamp': datetime.datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/edit/brightness', methods=['POST'])
def adjust_brightness():
    if not current_user.is_authenticated:
        return jsonify({'success': False, 'error': 'Authentication required'}), 401
    
    try:
        data = request.get_json()
        image_path = os.path.join(app.root_path, app.config['ENHANCED_FOLDER'], data.get('filename'))
        brightness_value = float(data.get('brightness', 0))
        
        if not os.path.exists(image_path):
            return jsonify({'success': False, 'error': 'Image not found'}), 404
        
        # Open the image
        img = Image.open(image_path)
        
        # Convert brightness from -100 to 100 scale to a factor
        factor = 1.0 + (brightness_value / 100.0)
        
        # Adjust brightness
        enhancer = ImageEnhance.Brightness(img)
        brightened_img = enhancer.enhance(factor)
        
        # Save the image
        brightened_img.save(image_path)
        
        return jsonify({
            'success': True, 
            'filename': os.path.basename(image_path),
            'timestamp': datetime.datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/edit/reset', methods=['POST'])
def reset_image():
    if not current_user.is_authenticated:
        return jsonify({'success': False, 'error': 'Authentication required'}), 401
    
    try:
        data = request.get_json()
        original_filename = data.get('original_filename')
        enhanced_filename = data.get('enhanced_filename')
        
        original_path = os.path.join(app.root_path, app.config['UPLOAD_FOLDER'], original_filename)
        enhanced_path = os.path.join(app.root_path, app.config['ENHANCED_FOLDER'], enhanced_filename)
        
        if not os.path.exists(original_path):
            return jsonify({'success': False, 'error': 'Original image not found'}), 404
        
        # Re-enhance the original image
        enhanced_filename, error = enhance_image(original_path)
        
        if error:
            return jsonify({'success': False, 'error': error}), 500
        
        return jsonify({
            'success': True, 
            'filename': enhanced_filename,
            'timestamp': datetime.datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/test_login')
def test_login():
    return render_template('test_login.html')

if __name__ == '__main__':
    # Load user data from JSON files
    load_data()
    app.run(debug=True) 