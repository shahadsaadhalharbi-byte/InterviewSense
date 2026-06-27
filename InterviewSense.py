
from flask import Flask, render_template, redirect, url_for, request, session, jsonify
import sqlite3
import os
from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from deepface import DeepFace
import cv2
import time
import assemblyai as aai
import sounddevice as sd
from scipy.io.wavfile import write
import tempfile
import base64
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('all-MiniLM-L6-v2')
import re


class Database:
    DATABASE = 'database.db'
    def __init__(self, database_path):
        self.database_path = database_path
        
    def get_connection(self):
        conn = sqlite3.connect(self.database_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def authenticate_user(self, email, password):
        conn = self.get_connection()
        user = conn.execute('SELECT * FROM users WHERE email = ? AND password = ?', 
                           (email, password)).fetchone()
        conn.close()
        return user
    
    def create_user(self, name, email, password):
        conn = self.get_connection()
        try:
            conn.execute('INSERT INTO users (name, email, password) VALUES (?, ?, ?)', 
                        (name, email, password))
            conn.commit()
            conn.close()
            return True
        except sqlite3.IntegrityError:
            conn.close()
            return False
    
    def get_user_by_id(self, user_id):
        conn = self.get_connection()
        user = conn.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
        conn.close()
        return user
    
    def update_user_profile(self, user_id, name, major):
        conn = self.get_connection()
        try:
            conn.execute('UPDATE users SET name = ?, major = ? WHERE id = ?', 
                        (name, major, user_id))
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Error updating user profile: {e}")
            conn.close()
            return False
            
    def save_contact(self, name, email, message):
        conn = self.get_connection()
        conn.execute('INSERT INTO contacts (name, email, message) VALUES (?, ?, ?)', 
                    (name, email, message))
        conn.commit()
        conn.close()

class ChatGPT:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)
    def generate_questions(self, major):
        timestamp = int(time.time())
        prompt = (
        f"Give  2 technical interview questions for {major}. "
        "Each question must have a clear and correct answer, without requiring code or calculations. "
        "Each question should be in one sentence only. No explanations or extra text. "
        "Avoid repetition in wording or concept. "
        f"Make them different every time. Session ID: {timestamp}"
    )

        response = self.client.chat.completions.create(
            model="gpt-4o-mini"  ,      
            messages=[{"role": "user", "content": prompt}],
            temperature=1.5 
        )
        content = response.choices[0].message.content
        questions = []
        for line in content.split('\n'):
            if any(line.strip().startswith(f"{i}.") for i in range(1, 8)):
                q = line.split(".", 1)[1].strip()
                if q:
                    questions.append(q)
        return questions
    
    def get_ideal_answer(self, question):
       prompt = (
        f"Give a concise but comprehensive answer to this technical interview question: {question}. "
        "Focus on key concepts and critical information rather than strict terminology. "
        "Write the answer as plain text, no code unless absolutely necessary. "
        "Limit the response to one clear and well-structured paragraph only."
        )
       response = self.client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
       return response.choices[0].message.content

class ASR:
    ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")
    def __init__(self, api_key):
        aai.settings.api_key = api_key
        self.transcriber = aai.Transcriber()

    def record_audio(self, duration=10, fs=44100):
        audio_data = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
        sd.wait()
        temp_wav_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
        write(temp_wav_path, fs, audio_data)
        return temp_wav_path, audio_data, fs
    
    def transcribe(self, audio_path):
        transcript = self.transcriber.transcribe(audio_path)
        return transcript.text if transcript.text else "No audio detected"    
    
    def record_and_transcribe(self, duration=15):
        try:
            audio_path, _, _ = self.record_audio(duration=duration)
            transcript = self.transcribe(audio_path)
            return {
                "success": True, 
                "text": transcript.strip() if transcript.strip() else "No answer provided."
            }
        except Exception as e:
            print(f"Error in record_and_transcribe: {e}")
            return {"success": False, "text": "No answer provided."}
    
    def record_question_answer(self, index):
        try:
            print(f"Recording answer for question {index}...")
            audio_path, _, _ = self.record_audio(duration=15)
            transcript = self.transcribe(audio_path)
            text = transcript.strip() if transcript.strip() else "No answer provided."
            return {"success": True, "text": text}
        except Exception as e:
            print(f"Error in record_question_answer: {e}")
            return {"success": False, "text": "No answer provided."}
    
    def handle_record_route(self):
        """Flask route handler for general recording."""
        try:
            result = self.record_and_transcribe(duration=60)
            return jsonify(result)
        except Exception as e:
            print("Recording error:", e)
            return jsonify({"success": False})

class FacialEmotionRecognition:
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.cap = None
        self.counts = {
            "happy": 0, "sad": 0, "angry": 0, "surprise": 0,
            "fear": 0, "neutral": 0, "disgust": 0
        }
        self.total = 0
        self.running = True
        # Load the Haar cascade for face detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def open_camera(self):
        if not self.cap:
            self.cap = cv2.VideoCapture(0)
        return self.cap.isOpened() 

    def detect_face(self, img):
        """
        Detects if there is a face in the image using OpenCV's Haar Cascade
        Returns True if a face is detected, False otherwise
        """
        if img is None:
            return False
            
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the image
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        # Return True if at least one face is detected
        return len(faces) > 0

    def analyze_frame(self, request, session):
        # First check if camera is actually enabled in the session
        if not session.get("camera_enabled") == "on":
            return jsonify({"error": "Camera not enabled", "camera_status": "off"})
        
        try:
            current_index = request.args.get('index', default=0, type=int)
            image_data = request.get_data().decode('utf-8')
            session_key = f'emotion_q{current_index}'
            
            # Initialize the emotion counts if not already present
            if session_key not in session:
                session[session_key] = {
                    "happy": 0, "sad": 0, "angry": 0, "surprise": 0,
                    "fear": 0, "neutral": 0, "disgust": 0,
                    "face_detected_count": 0,  # Track number of frames with faces detected
                    "total_frames": 0          # Track total frames analyzed
                }
            
            try:
                if image_data.startswith('data:image/jpeg;base64,'):
                    image_data = image_data.split(',')[1]

                image_bytes = base64.b64decode(image_data)
                nparr = np.frombuffer(image_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                # Update total frames count
                session[session_key]["total_frames"] = session[session_key].get("total_frames", 0) + 1
                
                # Detect if there's a face in the image first
                face_detected = self.detect_face(img)
                
                if face_detected:
                    # Update face detection count
                    session[session_key]["face_detected_count"] = session[session_key].get("face_detected_count", 0) + 1
                    
                    # Only analyze emotions if a face is detected
                    try:
                        result = DeepFace.analyze(img, actions=["emotion"], detector_backend="opencv", enforce_detection=False, silent=True)
                        emotion = result[0]['dominant_emotion']
                        
                        # Update emotion count
                        session[session_key][emotion] += 1
                        session.modified = True
                        
                        print(f"Detected emotion: {emotion} for question {current_index}")
                        
                        return jsonify({
                            "emotion": emotion,
                            "face_detected": True,
                            "counts": session[session_key]
                        })
                    except Exception as e:
                        print(f"DeepFace analysis error: {str(e)}")
                        pass
                
                # If we get here, either no face was detected or emotion analysis failed
                return jsonify({
                    "emotion": None,
                    "face_detected": face_detected,
                    "counts": session[session_key]
                })
                
            except Exception as e:
                print(f"Frame processing error: {str(e)}")
                pass
                
            return jsonify({
                "emotion": None,
                "face_detected": False,
                "counts": session[session_key]
            })
            
        except Exception as e:
            print(f"Frame processing error: {str(e)}")
            return jsonify({"error": f"Frame processing failed: {str(e)}"})

class Score:
    @staticmethod
    def calculate_similarity(user_answers, ideal_answers):
        try:
            # Get questions from session
            questions = session.get('questions', [])
            if not questions or len(questions) != len(user_answers):
                questions = [""] * len(user_answers)  # Fallback if questions not found

            # Check if any valid answers were provided
            valid_answers_count = 0
            valid_user_answers = []
            valid_ideal_answers = []

            for i, (user_answer, ideal_answer, question) in enumerate(zip(user_answers, ideal_answers, questions)):
                if not user_answer or user_answer in ["No audio detected", "No answer provided."]:
                    continue

                if len(user_answer.strip()) < 10:
                    continue

                # Check if answer is just repeating the question (or very similar)
                # First normalize both strings (lowercase, remove punctuation)
                def normalize_text(text):
                    return re.sub(r'[^\w\s]', '', text.lower())
                
                norm_question = normalize_text(question)
                norm_answer = normalize_text(user_answer)
                
                # If answer contains most of the question, it's likely a repetition
                question_words = set(norm_question.split())
                answer_words = set(norm_answer.split())
                if len(question_words) > 0:
                    overlap_ratio = len(question_words.intersection(answer_words)) / len(question_words)
                    
                    # If more than 70% of question words are in the answer, consider it a repetition
                    if overlap_ratio > 0.7:
                        print(f"Answer {i} appears to be repeating the question - skipping")
                        continue

                valid_answers_count += 1
                valid_user_answers.append(user_answer)
                valid_ideal_answers.append(ideal_answer)
                
            if valid_answers_count == 0:
                print("No valid answers detected - score is 0")
                return 0

            # Encode answers into embeddings
            user_embeddings = model.encode(valid_user_answers, convert_to_tensor=True)
            ideal_embeddings = model.encode(valid_ideal_answers, convert_to_tensor=True)

            # Compute cosine similarities
            similarities = []
            for user_emb in user_embeddings:
                cosine_sim = util.pytorch_cos_sim(user_emb, ideal_embeddings)
                similarities.append(cosine_sim.max().item())

            return (sum(similarities) / len(similarities)) * 100 if similarities else 0

        except Exception as e:
            print(f"Error in calculate_similarity: {e}")
            return 0


    @staticmethod
    def aggregate_emotions(emotion_data):
        combined_emotions = {
            "happy": 0, "sad": 0, "angry": 0, "surprise": 0,
            "fear": 0, "neutral": 0, "disgust": 0
        }
        
        # Track face detection metrics across all questions
        total_frames = 0
        total_frames_with_face = 0
        
        for emotion_counts in emotion_data:
            # Add face detection metrics if available
            if "total_frames" in emotion_counts and "face_detected_count" in emotion_counts:
                total_frames += emotion_counts["total_frames"]
                total_frames_with_face += emotion_counts["face_detected_count"]
        
        # Calculate face detection rate
        face_detection_rate = 0
        if total_frames > 0:
            face_detection_rate = (total_frames_with_face / total_frames) * 100
        
        # CRITICAL CHECK: If face detection rate is 0% or very low (threshold), return no emotions
        if face_detection_rate < 5 and total_frames > 5:  # Less than 5% face detection and enough samples
            return {
                "error": "No face detected", 
                "face_detection_rate": 0
            }, True
        
        # Only if we have enough face detections, process the emotion data
        if total_frames_with_face > 0:
            # Now process emotions only from frames that had faces
            for emotion_counts in emotion_data:
                # Add emotion counts - but only if we have face detection data
                if "face_detected_count" in emotion_counts and emotion_counts["face_detected_count"] > 0:
                    for emotion, count in emotion_counts.items():
                        if emotion in combined_emotions and isinstance(count, int):
                            combined_emotions[emotion] += count
            
            # Calculate emotion percentages if we have valid emotion data
            emotion_sum = sum(combined_emotions.values())
            if emotion_sum > 0:
                # Step 1: Get raw percentages
                raw_percentages = {
                    k: (v / emotion_sum) * 100 for k, v in combined_emotions.items()
                }

                # Step 2: Round down all values
                floored = {k: int(p) for k, p in raw_percentages.items()}

                # Step 3: Calculate remaining points to reach 100
                diff = 100 - sum(floored.values())

                # Step 4: Distribute remaining points by largest decimal parts
                decimals = sorted(
                    raw_percentages.items(),
                    key=lambda x: x[1] - int(x[1]),
                    reverse=True
                )

                for i in range(diff):
                    if i < len(decimals):
                        floored[decimals[i][0]] += 1

                # Step 5: Sort results from highest to lowest
                final_emotions = dict(sorted(floored.items(), key=lambda item: item[1], reverse=True))
                
                # Add face detection rate
                final_emotions["face_detection_rate"] = round(face_detection_rate, 1)
                
                return final_emotions, False
        
        # If we get here, we didn't have enough valid emotion data
        return {
            "error": "Insufficient face detection", 
            "face_detection_rate": round(face_detection_rate, 1)
        }, True
    
class Application:
    SECRET_KEY = os.getenv("FLASK_SECRET_KEY", "dev-secret-key")

    def __init__(self):
        self.app = Flask(__name__)
        self.app.secret_key = Application.SECRET_KEY
        
        # Initialize components
        self.db = Database(Database.DATABASE)
        self.chatgpt = ChatGPT(ChatGPT.OPENAI_API_KEY)
        self.asr = ASR(ASR.ASSEMBLYAI_API_KEY)
        self.fer = FacialEmotionRecognition()
        
        # Register routes
        self.register_routes()
        
    def register_routes(self):
        # Main routes
        self.app.route('/')(self.home)
        self.app.route('/get-started')(self.get_started)
        self.app.route('/about-us')(self.about)
        self.app.route('/login', methods=['GET', 'POST'])(self.login)
        self.app.route('/signup', methods=['GET', 'POST'])(self.signup)
        self.app.route('/form-page')(self.form_page)
        self.app.route('/frequentlyAskedQuestions')(self.faq)
        self.app.route('/profile')(self.profile)
        self.app.route('/update-profile', methods=['POST'])(self.update_profile)
        
        # Interview functionality routes
        self.app.route('/inter_view', methods=['POST'])(self.inter_view)
        self.app.route('/record', methods=['POST'])(self.record_route)
        self.app.route('/record-question', methods=['POST'])(self.record_question)
        self.app.route('/submit-interview', methods=['POST'])(self.submit_interview)
        self.app.route('/analyze-frame', methods=['POST'])(self.analyze_frame_route)
        self.app.route('/score-page')(self.score_page)
        
        # Error and utility routes
        self.app.route('/something-went-wrong')(self.error)
        self.app.route('/page-not-found')(self.page_404)
        self.app.route('/logout')(self.logout)
    
    # Route handlers
    def home(self):
        return render_template('homePage.html')
    
    def get_started(self):
        if 'user_id' in session:
            return render_template('form-page.html')
        return redirect(url_for('login'))
    
    def login(self):
        if request.method == 'POST':
            email = request.form['email']
            password = request.form['password']
            user = self.db.authenticate_user(email, password)
            if user:
                session['user_id'] = user['id']
                return redirect(url_for('get_started'))
            return render_template('login.html', error="Invalid email or password. Please try again.")
        return render_template('login.html')
    
    def signup(self):
        if request.method == 'POST':
            name = request.form['name']
            email = request.form['email']
            password = request.form['password']
            if self.db.create_user(name, email, password):
                return redirect(url_for('login'))
            return render_template('signup.html', error="Email already exists. Please use a different email address.")
        return render_template('signup.html')
    
    def form_page(self):
        return render_template('form-page.html')
    
    def about(self):
        return render_template('about-us.html')
    
    def faq(self):
        return render_template('frequentlyAskedQuestions.html')
    
    
    def profile(self):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        
        user = self.db.get_user_by_id(session['user_id'])
        return render_template('profile.html', user=user)
    
    def update_profile(self):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        
        user_id = session['user_id']
        name = request.form.get('name', '')
        major = request.form.get('major', '')
        
        # Validate inputs
        if not name:
            return render_template('profile.html', 
                                  user=self.db.get_user_by_id(user_id),
                                  error="Name cannot be empty")
        
        # Update user profile (without password change)
        if self.db.update_user_profile(user_id, name, major):
            return render_template('profile.html', 
                                  user=self.db.get_user_by_id(user_id),
                                  success="Profile updated successfully")
        else:
            return render_template('profile.html', 
                                  user=self.db.get_user_by_id(user_id),
                                  error="An error occurred while updating your profile")
    
    
    def inter_view(self):
        try:
            # Get form data safely
            major = request.form.get("major")
            camera_enabled = request.form.get("camera") == "on"
            mic_enabled = request.form.get("mic") == "on"

            # Validate microphone is required
            if not mic_enabled:
                print("Microphone is not enabled. Redirecting to error.")
                return redirect(url_for('error'))

            # If no major selected (fallback)
            if not major:
                major = "Computer Science"

            # Generate clean interview questions
            questions = self.chatgpt.generate_questions(major)

            # Store in session for later use
            session['questions'] = questions
            session['camera_enabled'] = "on" if camera_enabled else "off"

            # Pass questions to the page
            return render_template("interview.html", questions=questions, camera_enabled=camera_enabled)
        except Exception as e:
            print(f"Error in pop-out: {e}")
            return redirect(url_for('error'))
    


    def record_question(self):
        try:
            index = int(request.form.get('index', -1))
            if index < 0:
                return jsonify({"success": False, "text": "Invalid index"})

            # Initialize FER if camera is enabled
            fer = None
            if session.get("camera_enabled") == "on":
                fer = FacialEmotionRecognition(enabled=True)
                fer.open_camera()

            # Use the reorganized ASR class method
            result = self.asr.record_question_answer(index)
        
            
            
            # Process video frames during recording
            start_time = time.time()
            while time.time() - start_time < 15:  # 15 seconds matches audio duration
                if fer and fer.cap:
                    ret, frame = fer.cap.read()
                    if ret:
                        fer.analyze_frame(frame)
                time.sleep(0.1)  # Avoid CPU overload

            # Store RAW emotion counts for this question
            if fer:
                session[f'emotion_q{index}'] = fer.counts
                fer.close()      
            
            # Return the transcription result
            return jsonify(result)
        
        except Exception as e:
            print(f"Error in record-question: {e}")
            return jsonify({"success": False, "text": "No answer provided."})
        
    
    def analyze_frame_route(self):
        # This is now just a route handler that calls the method in FacialEmotionRecognition
        return self.fer.analyze_frame(request, session)

    def record_route(self):
        # Now just a route handler that calls the method in ASR
        return self.asr.handle_record_route()
    

    def submit_interview(self):
        try:
            data = request.get_json()
            user_answers = data.get('answers', [])
            questions = session.get('questions', [])

            print("Questions:", questions)
            print("User answers:", user_answers)


            # Calculate answer similarity scores
            ideal_answers = [self.chatgpt.get_ideal_answer(q) for q in questions]
            print("Ideal answers:", ideal_answers) 
            similarity_score = Score.calculate_similarity(user_answers, ideal_answers)
            print("Similarity score:", similarity_score)

            # Check if camera was enabled
            camera_enabled = session.get('camera_enabled') == "on"
            
            # If camera was disabled, set appropriate values
            if not camera_enabled:
                session['final_score'] = similarity_score
                session['emotion_result'] = {"camera_disabled": True}
                session['no_face_detected'] = True
                
                return jsonify({
                    "success": True, 
                    "score": similarity_score, 
                    "emotions": {"camera_disabled": True}
                })

            # Otherwise process emotions as usual
            emotion_data = []
            for i in range(len(questions)):
                q_emotions = session.get(f'emotion_q{i}', {})
                print(f"Emotion data for question {i}:", q_emotions)
                
                # Add default empty structure if no data was collected
                if not q_emotions:
                    q_emotions = {
                        "happy": 0, "sad": 0, "angry": 0, "surprise": 0,
                        "fear": 0, "neutral": 0, "disgust": 0,
                        "total_frames": 0, "face_detected_count": 0
                    }
                
                emotion_data.append(q_emotions)
                
            # Process emotions
            final_emotions, no_face_detected = Score.aggregate_emotions(emotion_data)
            print("Final emotions:", final_emotions)
            print("No face detected:", no_face_detected)

            session['final_score'] = similarity_score
            session['emotion_result'] = final_emotions
            session['no_face_detected'] = no_face_detected
            
            return jsonify({
                "success": True, 
                "score": similarity_score, 
                "emotions": final_emotions
            })
        except Exception as e:
            print(f"Error in submit-interview: {str(e)}")
            return jsonify({"success": False, "error": str(e)})
    
    def score_page(self):
        emotions = session.pop('emotion_result', {})
        score = float(session.pop('final_score', 0))
        no_face_detected = session.pop('no_face_detected', False)
        keys_to_remove = [key for key in session.keys() if key.startswith("emotion_q")]
        for key in keys_to_remove:
            session.pop(key)
        return render_template('score-page.html', emotions=emotions, similarity_score=int(score), no_face=no_face_detected)

    def error(self):
        return render_template('somethingWrong.html')
    
    def page_404(self):
        return render_template('pageNotFound.html')
    
    def logout(self):
        session.clear()
        return redirect(url_for('home'))

# Create Flask app instance
app = Application().app

if __name__ == '__main__':
    app.run(debug=True)