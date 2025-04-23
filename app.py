# --- START OF FILE app.py ---

from flask import Flask, request, jsonify, render_template, redirect, url_for, session, Response, stream_with_context
import base64
from io import BytesIO
from PIL import Image
import files.api.predict as predict_lib # Assuming predict.py is in files/api/
import tensorflow as tf
import os
import datetime
import uuid
import traceback
import tempfile
import re # Import regular expressions for filename parsing
import glob # Import glob for file pattern matching

# --- Imports for Spoken-to-Signed Feature ---
# ... (keep existing imports)
import sys
# import subprocess # No longer needed as pose_to_video is removed
import json
from faster_whisper import WhisperModel
from pose_format import Pose
from pose_format.pose_visualizer import PoseVisualizer # Keep for GIF
# Ensure the gloss_to_pose module is on the Python path.
# You might need to adjust this path depending on your exact project structure
# Assuming it's adjacent to the main script or installable
try:
    sys.path.append(os.path.join(os.getcwd(), "spoken-to-signed-translation", "spoken_to_signed", "gloss_to_pose"))
    from concatenate import concatenate_poses
except ImportError as e:
    print(f"Warning: Could not import 'concatenate_poses'. Ensure 'gloss_to_pose' module is accessible: {e}")
    # Define a dummy function if needed, or let it fail later
    def concatenate_poses(poselist):
        print("WARNING: Using dummy concatenate_poses function!")
        if not poselist: return None # Or raise error
        # Basic concatenation might involve just taking the first pose or raising an error
        # For simplicity, let's just return the first pose if it exists
        return poselist[0]
# --- End Imports for Spoken-to-Signed Feature ---


# --- Database & Auth Setup ---
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, current_user, login_required
from werkzeug.security import generate_password_hash, check_password_hash
from flask_migrate import Migrate # For database migrations

# --- Configuration ---
# Configure the template folder relative to the app.py file location
TEMPLATE_FOLDER = os.path.join(os.path.dirname(__file__), 'files', 'templates')
# Point static folder to 'files/static'
STATIC_FOLDER_PATH = os.path.join(os.path.dirname(__file__), 'files', 'static')
app = Flask(__name__, template_folder=TEMPLATE_FOLDER, static_folder=STATIC_FOLDER_PATH)
# Flask will automatically map the URL path '/static' to the contents of STATIC_FOLDER_PATH

# **IMPORTANT: Set a strong SECRET_KEY for production!**
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'a_very_insecure_default_key_for_dev_12345') # Added more default chars for basic security

# --- MySQL (XAMPP) Database Configuration ---
# !!! REPLACE placeholders with your actual MySQL details !!!
db_username = os.environ.get('DB_USER', 'sign2speak_user')      # Use environment variable or default
db_password = os.environ.get('DB_PASS', 'Sv227199')   # Use environment variable or default
db_host = os.environ.get('DB_HOST', 'localhost')              # Use environment variable or default
db_port = os.environ.get('DB_PORT', '3306')                   # Use environment variable or default
database_name = os.environ.get('DB_NAME', 'sign2speak_db')     # Use environment variable or default

app.config['SQLALCHEMY_DATABASE_URI'] = (
    f'mysql+mysqlconnector://{db_username}:{db_password}@{db_host}:{db_port}/{database_name}'
)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize extensions
db = SQLAlchemy(app)
migrate = Migrate(app, db)
login_manager = LoginManager(app)
login_manager.login_view = 'signin_page'
login_manager.login_message = "Please log in to access this page."
login_manager.login_message_category = 'info'

# --- User Model ---
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    full_name = db.Column(db.String(100), nullable=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def __repr__(self):
        return f'<User {self.email}>'

# User loader function for Flask-Login
@login_manager.user_loader
def load_user(user_id):
    try:
        return User.query.get(int(user_id))
    except (TypeError, ValueError):
        return None

# --- Path & Model Config ---
BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "files", "model")
MODEL_NAME = "asl_model.keras"
model_path = os.path.join(MODEL_DIR, MODEL_NAME)
TMP_DIR = os.path.join(BASE_DIR, "files", "api", "tmp") # Base directory for sign-to-text frames
TEMP_DIR_AUDIO = os.path.join(BASE_DIR, "temp") # Used for audio and temp pose

# --- Create Directories ---
os.makedirs(TMP_DIR, exist_ok=True) # Ensure base temp directory exists
os.makedirs(TEMP_DIR_AUDIO, exist_ok=True)
# Ensure 'videos' subdirectory exists in static folder (for GIF output)
VIDEOS_DIR = os.path.join(STATIC_FOLDER_PATH, "videos")
os.makedirs(VIDEOS_DIR, exist_ok=True)
print(f"Directory for generated GIFs/videos created/exists at: {VIDEOS_DIR}")

# Base ASL pose directory path check (corrected path)
ASL_POSE_DIR = os.path.join(BASE_DIR, "files", "static", "asl")
if not os.path.isdir(ASL_POSE_DIR):
    print(f"WARNING: ASL Pose directory not found at {ASL_POSE_DIR}. Spoken-to-signed feature will fail.")
else:
    print(f"ASL Pose directory found at: {ASL_POSE_DIR}")

# --- Keras ASL Model Loading (Sign-to-Text) ---
loaded_model = None
try:
    if os.path.exists(model_path):
        loaded_model = tf.keras.models.load_model(model_path, compile=False)
        print(f"ASL Model loaded successfully from {model_path}")
    else:
        print(f"Error: ASL Model file not found at {model_path}. Prediction endpoint will not work.")
except Exception as e:
    print(f"Error loading ASL model from {model_path}: {e}. Prediction endpoint will not work.")

# --- Faster Whisper Model Loading (for Spoken-to-Signed GIF Feature) ---
spoken_to_signed_model_size = "distil-small.en" # Smaller, faster model
spoken_to_signed_model = None
try:
    # Consider trying 'cpu' and 'int8' for wider compatibility if 'auto' fails
    spoken_to_signed_model = WhisperModel(spoken_to_signed_model_size, device="cpu", compute_type="int8")
    print(f"Faster Whisper model '{spoken_to_signed_model_size}' loaded (cpu, int8).")
except Exception as e:
    print(f"Error loading faster-whisper model '{spoken_to_signed_model_size}': {e}")
    print("Spoken-to-signed GIF generation endpoint /process will likely fail.")

# --- Helper function to pass auth status to templates ---
@app.context_processor
def inject_user():
    return dict(current_user=current_user)

# --- Web Page Routes ---
@app.route("/")
def landing_page():
    """Serves the main landing page (index.html)."""
    try: return render_template("index.html")
    except Exception as e: print(f"Error rendering landing page: {e}"); traceback.print_exc(); return "Error loading page.", 500

@app.route("/signin")
def signin_page():
    """Serves the sign-in page (signin.html). Redirects if already logged in."""
    if current_user.is_authenticated: return redirect(url_for('main_app_page'))
    try: return render_template("signin.html")
    except Exception as e: print(f"Error rendering signin page: {e}"); traceback.print_exc(); return "Error loading page.", 500

@app.route("/signup")
def signup_page():
    """Serves the sign-up page (signup.html). Redirects if already logged in."""
    if current_user.is_authenticated: return redirect(url_for('main_app_page'))
    try: return render_template("signup.html")
    except Exception as e: print(f"Error rendering signup page: {e}"); traceback.print_exc(); return "Error loading page.", 500

@app.route("/register")
def register_page():
    """Redirects /register to the signup page for consistency."""
    return redirect(url_for('signup_page'))

@app.route("/app")
@login_required # User MUST be logged in to access this page
def main_app_page():
    """Serves the main Sign 2 Speak application page (app.html). Requires login."""
    try: return render_template("app.html")
    except Exception as e: print(f"Error rendering main app page: {e}"); traceback.print_exc(); return "Error loading application.", 500

# --- API Routes ---

# --- Authentication API ---
@app.route("/api/signup", methods=['POST'])
def api_signup():
    """Handles user registration via API."""
    if current_user.is_authenticated: return jsonify({"status": "error", "message": "Already logged in."}), 400
    if not request.is_json: return jsonify({"status": "error", "message": "Request must be JSON."}), 415
    data = request.get_json(); email = data.get('email'); password = data.get('password'); full_name = data.get('fullName')
    if not all([email, password, full_name]): return jsonify({"status": "error", "message": "Missing email, password, or full name."}), 400
    if len(password) < 8: return jsonify({"status": "error", "message": "Password must be at least 8 characters."}), 400
    # Use case-insensitive query for email check
    if User.query.filter(User.email.ilike(email)).first(): return jsonify({"status": "error", "message": "Email address already registered."}), 409
    try: new_user = User(email=email, full_name=full_name); new_user.set_password(password); db.session.add(new_user); db.session.commit(); print(f"User registered successfully: {email}"); return jsonify({"status": "success", "message": "Account created successfully. Please sign in."}), 201
    except Exception as e: db.session.rollback(); print(f"Error during signup: {e}"); traceback.print_exc(); return jsonify({"status": "error", "message": "Registration failed due to a server error."}), 500

@app.route("/api/signin", methods=['POST'])
def api_signin():
    """Handles user login via API."""
    if current_user.is_authenticated: return jsonify({"status": "error", "message": "Already logged in."}), 400
    if not request.is_json: return jsonify({"status": "error", "message": "Request must be JSON."}), 415
    data = request.get_json(); email = data.get('email'); password = data.get('password'); remember = data.get('rememberMe', False)
    if not all([email, password]): return jsonify({"status": "error", "message": "Missing email or password."}), 400
    # Use case-insensitive query for email lookup
    user = User.query.filter(User.email.ilike(email)).first()
    if user and user.check_password(password): login_user(user, remember=remember); print(f"User logged in successfully: {email}"); return jsonify({"status": "success", "message": "Login successful."})
    else: print(f"Login failed for email: {email}"); return jsonify({"status": "error", "message": "Invalid email or password."}), 401

@app.route("/api/logout")
@login_required # Ensures only logged-in users can trigger logout
def api_logout():
    """Handles user logout via API."""
    if current_user.is_authenticated: # Check is slightly redundant due to @login_required but good practice
        print(f"User logging out: {current_user.email}")
        logout_user()
        # Redirect to sign-in page after logout
        return redirect(url_for('signin_page'))
    else:
        # Should not happen if @login_required works, but handle defensively
        return redirect(url_for('signin_page'))


@app.route("/api/check_auth", methods=['GET'])
def check_auth_status():
    """Allows frontend to check if a user is logged in and get basic info."""
    if current_user.is_authenticated: return jsonify({"isLoggedIn": True, "user": {"email": current_user.email, "fullName": current_user.full_name}})
    else: return jsonify({"isLoggedIn": False})

# --- Prediction API (Sign-to-Text) ---
@app.route("/upload_frame", methods=['POST'])
#@login_required # Uncomment if login is required
def upload_frame():
    """Handles uploading a single frame for prediction. Saves with UUID."""
    if not request.is_json: return jsonify({"status": "error", "message": "Request must be JSON."}), 415
    try:
        req_token = str(uuid.uuid4())
        data = request.get_json()
        if not data or 'image' not in data: return jsonify({"status": "error", "message": "Missing 'image' data."}), 400
        image_data_url = data.get('image')
        if not isinstance(image_data_url, str) or not image_data_url.startswith('data:image/png;base64,'): return jsonify({"status": "error", "message": "Invalid image data format."}), 400
        try: header, image_data_b64 = image_data_url.split(",", 1)
        except ValueError: return jsonify({"status": "error", "message": "Malformed image data URL."}), 400
        try: image_binary = base64.b64decode(image_data_b64)
        except Exception as e: return jsonify({"status": "error", "message": f"Base64 decoding error: {e}"}), 400
        try: image = Image.open(BytesIO(image_binary)).convert("RGB")
        except Exception as e: return jsonify({"status": "error", "message": f"Error opening image data: {e}"}), 400

        # Save temporarily with UUID in the base TMP_DIR
        save_filename = f"captured_frame-{req_token}.png"; save_path = os.path.join(TMP_DIR, save_filename)
        try: image.save(save_path, format="PNG")
        except Exception as e: print(f"Error saving frame: {e}"); traceback.print_exc(); return jsonify({"status": "error", "message": f"Error saving image file: {e}"}), 500

        # Return the token (UUID)
        return jsonify({"status": "success", "message": "Frame uploaded.", "token": req_token})
    except Exception as e: print(f"Unexpected error in /upload_frame: {e}"); traceback.print_exc(); return jsonify({"status": "error", "message": "Internal server error."}), 500


@app.route("/predict", methods=["POST"])
#@login_required # Uncomment if login is required
def predict():
    """
    Performs prediction on an uploaded frame using its token.
    If prediction is successful (A-Z), moves the temp file to a subdirectory
    named after the prediction (e.g., A/) and renames it to
    {PredictedLetter}{SequenceNumber}.png. Otherwise, deletes the temp file.
    """
    global loaded_model
    if loaded_model is None: return jsonify({"status": "error", "message": "ASL Model not loaded."}), 503
    if not request.is_json: return jsonify({"status": "error", "message": "Request must be JSON."}), 415

    temp_image_path = None # Initialize to None
    try:
        data = request.get_json()
        if not data or 'token' not in data: return jsonify({"status": "error", "message": "Missing 'token'."}), 400
        image_token = data.get('token')
        # Basic validation for UUID-like tokens
        if not isinstance(image_token, str) or not re.match(r'^[a-f0-9\-]{36}$', image_token, re.IGNORECASE):
            return jsonify({"status": "error", "message": "Invalid token format."}), 400

        # Construct the initial temporary file path in the base TMP_DIR
        temp_image_filename = f"captured_frame-{image_token}.png"
        temp_image_path = os.path.join(TMP_DIR, temp_image_filename)

        if not os.path.exists(temp_image_path) or not os.path.isfile(temp_image_path):
            print(f"Pred error: File not found {temp_image_path}")
            return jsonify({"status": "error", "message": "Image not found."}), 404

        # --- Perform Prediction ---
        landmark_data = predict_lib.get_marks(temp_image_path)
        if landmark_data is None:
            # No hand detected or error getting landmarks - Delete temp file
            print(f"No landmarks detected for {temp_image_filename}. Deleting.")
            if os.path.exists(temp_image_path):
                try: os.remove(temp_image_path)
                except OSError as e: print(f"Warn: Cleanup failed for {temp_image_path}: {e}")
            return jsonify({"status": "success", "message": "No hand detected", "result": ""})

        prediction_result = predict_lib._predict(landmark_data, loaded_model)

        # --- Handle Renaming/Moving or Deletion ---
        if prediction_result and 'A' <= prediction_result <= 'Z':
            # Valid prediction - Move and Rename the file into a subdirectory
            try:
                # --- Create Subdirectory ---
                target_subdir = os.path.join(TMP_DIR, prediction_result)
                os.makedirs(target_subdir, exist_ok=True) # Create if it doesn't exist

                # --- Find the next sequence number within the subdirectory ---
                sequence_number = 1
                # Search pattern inside the target subdirectory
                pattern = os.path.join(target_subdir, f"{prediction_result}[0-9]*.png")
                existing_files = glob.glob(pattern)
                if existing_files:
                    max_num = 0
                    for f in existing_files:
                        basename = os.path.basename(f)
                        # Extract number (handle cases like A.png, A1.png, A10.png)
                        match = re.match(rf"^{prediction_result}(\d+)\.png$", basename)
                        if match:
                            num = int(match.group(1))
                            if num > max_num:
                                max_num = num
                    sequence_number = max_num + 1

                # --- Construct New Path inside Subdirectory ---
                new_filename = f"{prediction_result}{sequence_number}.png"
                new_image_path = os.path.join(target_subdir, new_filename)

                print(f"Moving and Renaming {temp_image_filename} to {os.path.join(prediction_result, new_filename)}")
                os.rename(temp_image_path, new_image_path) # Move/Rename
                temp_image_path = None # Prevent deletion of original path in finally block

            except Exception as move_rename_err:
                print(f"Error moving/renaming file {temp_image_filename}: {move_rename_err}")
                # Fallback: Delete the original temp file if move/rename failed
                if temp_image_path and os.path.exists(temp_image_path):
                    try: os.remove(temp_image_path)
                    except OSError as e: print(f"Warn: Cleanup failed for {temp_image_path} after move/rename error: {e}")
                # Still return the prediction, but log the error
        else:
            # Invalid or empty prediction - Delete the original temp file
            print(f"Invalid/Empty prediction ('{prediction_result}') for {temp_image_filename}. Deleting.")
            if os.path.exists(temp_image_path):
                try: os.remove(temp_image_path)
                except OSError as e: print(f"Warn: Cleanup failed for {temp_image_path}: {e}")
            temp_image_path = None # Prevent potential double-deletion attempt

        # Return successful prediction result
        return jsonify({"status": "success", "message": "Prediction successful", "result": prediction_result})

    except Exception as e:
        print(f"Error during prediction endpoint: {e}"); traceback.print_exc()
        # Attempt cleanup if temp_image_path was set and still exists
        if temp_image_path and os.path.exists(temp_image_path):
            try: os.remove(temp_image_path)
            except OSError as cleanup_err: print(f"Error cleaning up {temp_image_path} in exception handler: {cleanup_err}")
        return jsonify({"status": "error", "message": "Internal prediction error."}), 500


# --- REVISED: Spoken-to-Signed GIF Generation Stream (Using Transcription) ---

def stream_process_audio_file(audio_path):
    """
    Generator function: Transcribes audio, loads poses based on transcribed words,
    generates GIF preview, saves temp pose. NO VIDEO.
    """
    global spoken_to_signed_model
    if spoken_to_signed_model is None:
        yield f"data: {json.dumps({'step': 'error', 'error': 'Speech processing model not loaded.'})}\n\n"
        if os.path.exists(audio_path):
            try: os.remove(audio_path)
            except OSError as e: print(f"Error removing audio file {audio_path}: {e}")
        return

    temp_pose_path = None # For the temporary concatenated pose file
    gif_path = None # To store path for potential cleanup

    try:
        # STEP 1: Transcribe audio
        yield f"data: {json.dumps({'step': 'info', 'message': 'Transcribing audio...'})}\n\n"
        segments, info = spoken_to_signed_model.transcribe(audio_path, beam_size=5, language="en", condition_on_previous_text=False, vad_filter=True)
        merged_text = " ".join([segment.text for segment in segments]).strip()
        print("Transcribed text:", merged_text)
        yield f"data: {json.dumps({'step': 'transcription', 'transcription': merged_text or 'No speech detected.'})}\n\n"

        if not merged_text: # Handle case where transcription is empty
             yield f"data: {json.dumps({'step': 'info', 'message': 'Skipping pose generation due to empty transcription.'})}\n\n"
             return # Exit the generator

        # --- STEP 4: Load pose files based on *transcribed* words ---
        # Basic tokenization: split by space, convert to lowercase, remove punctuation
        tokens = [word.lower().strip(".,!?;:") for word in merged_text.split() if word.strip(".,!?;:")]

        if not tokens:
            yield f"data: {json.dumps({'step': 'info', 'message': 'No valid words found in transcription for pose lookup.'})}\n\n"
            return

        poselist = []
        yield f"data: {json.dumps({'step': 'info', 'message': f'Loading poses for: {" ".join(tokens)}'})}\n\n"
        print("Loading pose files for tokens:", tokens)
        missing_poses = [] # Keep track of missing ones

        for token in tokens:
            pose_file = os.path.join(ASL_POSE_DIR, f"{token}.pose") # Already lowercase
            print(f"Checking for pose file at: {pose_file}")

            if not os.path.isfile(pose_file):
                error_msg = f"Pose file not found for word '{token}'" # Simplified msg
                print(f"WARNING: {error_msg} at {pose_file}")
                missing_poses.append(token) # Add to missing list
                continue # Skip this token and try the next one
            else:
                 print(f"Pose file FOUND at: {pose_file}")

            try:
                with open(pose_file, "rb") as f: data = f.read()
                pose = Pose.read(data)
                poselist.append(pose)
                print(f"Successfully loaded pose for token: '{token}'")
            except Exception as e:
                 error_msg = f"Error reading pose file for '{token}' ({pose_file}): {e}"
                 print(f"ERROR: {error_msg}")
                 # Report error for this specific file but continue
                 yield f"data: {json.dumps({'step': 'warning', 'message': error_msg})}\n\n"
                 # Decide if you want to stop entirely on a read error:
                 # return

        # --- Report all missing poses at the end ---
        if missing_poses:
            yield f"data: {json.dumps({'step': 'warning', 'message': f'Missing pose files for: {", ".join(missing_poses)}'})}\n\n"

        # Check if *any* poses were successfully loaded
        if not poselist:
            yield f"data: {json.dumps({'step': 'error', 'error': 'No valid pose files could be loaded for the transcribed text.'})}\n\n"
            return

        # Concatenate poses
        yield f"data: {json.dumps({'step': 'info', 'message': 'Concatenating loaded poses...'})}\n\n"
        newpose = concatenate_poses(poselist)
        if newpose is None: # Check if concatenation failed
             yield f"data: {json.dumps({'step': 'error', 'error': 'Failed to concatenate poses.'})}\n\n"
             return


        # STEP 5: Generate a GIF preview of the pose.
        gif_filename = f"{uuid.uuid4()}.gif"
        gif_path = os.path.join(VIDEOS_DIR, gif_filename) # Save GIF in files/static/videos
        try:
            yield f"data: {json.dumps({'step': 'info', 'message': 'Generating GIF preview...'})}\n\n"
            visualizer = PoseVisualizer(newpose)
            visualizer.save_gif(gif_path, visualizer.draw())
            gif_url = url_for('static', filename=f'videos/{gif_filename}', _external=False)
            # Send GIF URL as the final successful step
            yield f"data: {json.dumps({'step': 'pose_gif', 'gif_filename': gif_filename, 'url': gif_url})}\n\n"
            print(f"GIF preview saved to {gif_path}")
        except Exception as e:
            error_msg = f"Error generating GIF: {e}"
            print(f"ERROR: {error_msg}")
            yield f"data: {json.dumps({'step': 'error', 'error': error_msg})}\n\n"
            return # Stop if GIF fails

        # STEP 6: Save the concatenated pose into a temporary file
        try:
            temp_pose_path = os.path.join(TEMP_DIR_AUDIO, f"{uuid.uuid4()}.pose")
            with open(temp_pose_path, "wb") as f: newpose.write(f)
            print(f"Temporary pose saved to {temp_pose_path} (will be cleaned up)")
        except Exception as e:
            print(f"Warning: Failed to save temporary pose file: {e}")
            # Don't yield an error for this, just log it


        # --- VIDEO GENERATION STEP REMOVED ---

    except Exception as e:
        print(f"Unexpected error in stream_process_audio_file: {e}"); traceback.print_exc()
        yield f"data: {json.dumps({'step': 'error', 'error': 'An unexpected server error occurred.'})}\n\n"
    finally:
        # --- Cleanup ---
        if temp_pose_path and os.path.exists(temp_pose_path):
            try: os.remove(temp_pose_path); print(f"Cleaned up temp pose file: {temp_pose_path}")
            except OSError as e_clean: print(f"Error cleaning up temp pose file '{temp_pose_path}': {e_clean}")
        if os.path.exists(audio_path):
            try: os.remove(audio_path); print(f"Cleaned up temp audio file: {audio_path}")
            except OSError as e_clean: print(f"Error cleaning up temp audio file '{audio_path}': {e_clean}")

        yield f"data: {json.dumps({'step': 'info', 'message': 'Processing finished.'})}\n\n"


@app.route("/process", methods=["POST"])
#@login_required
def process_audio_for_gif(): # Renamed function
    """
    Receives audio, saves temporarily, and starts the SSE stream for
    transcription and pose GIF generation based on transcribed words.
    """
    global spoken_to_signed_model
    if spoken_to_signed_model is None: return jsonify({"error": "Speech processing model not loaded."}), 503
    if "audio" not in request.files: return jsonify({"error": "No audio file provided."}), 400
    audio_file = request.files["audio"]
    if audio_file.filename == '': return jsonify({"error": "No selected audio file."}), 400

    # Determine suffix from original filename if possible, otherwise default
    _, suffix = os.path.splitext(audio_file.filename)
    if not suffix: suffix = ".webm" # Default if no extension found

    audio_path = None
    try:
        # Save audio temporarily (ensure TEMP_DIR_AUDIO exists)
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=TEMP_DIR_AUDIO) as tmp: audio_path = tmp.name
        audio_file.save(audio_path); print(f"Temporary audio saved to: {audio_path}")
    except Exception as e:
         print(f"Error saving temporary audio file: {e}")
         if audio_path and os.path.exists(audio_path):
             try: os.remove(audio_path)
             except Exception: pass
         return jsonify({"error": "Failed to save audio file."}), 500

    return Response(stream_with_context(stream_process_audio_file(audio_path)), mimetype="text/event-stream")

# --- Main Execution ---
if __name__ == '__main__':
    # Instructions for running:
    # 1. Ensure MySQL/XAMPP is running and database/user exist. Update credentials above or use ENV variables.
    # 2. Ensure virtual environment is activated.
    # 3. Set FLASK_APP=app.py (using Powershell: $env:FLASK_APP = "app.py" OR using bash: export FLASK_APP=app.py)
    # 4. Run DB migrations if needed: flask db init (once), flask db migrate -m "message", flask db upgrade
    # 5. Install dependencies: pip install -r requirements.txt (or manually install all listed imports)
    #    Ensure ffmpeg is installed system-wide and in PATH (for whisper).
    # 6. Ensure necessary lowercase `.pose` files exist in 'files/static/asl/' (e.g., 'hello.pose', 'world.pose')
    # 7. Run: flask run --host=0.0.0.0 --port=5000
    #    Use debug=True ONLY for development: flask run --host=0.0.0.0 --port=5000 --debug
    #    For production, use a proper WSGI server like Gunicorn or uWSGI.
    app.run(debug=False, host='0.0.0.0', port=5000) # Set debug=True for development, but False for stability

# --- END OF FILE app.py ---
