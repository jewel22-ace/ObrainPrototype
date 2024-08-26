from flask import Flask, request, jsonify, render_template, redirect, url_for, session
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import pandas as pd
from sentence_transformers import SentenceTransformer
import sqlite3
import numpy as np
import os

app = Flask(__name__)
app.secret_key = 'your_secret_key'
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

MODEL_PATH = 'models/all-MiniLM-L6-v2'
DB_PATH = 'embeddings.db'
UPLOAD_FOLDER = 'uploads'

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load the sentence transformer model from the local directory
model = SentenceTransformer(MODEL_PATH)

class User(UserMixin):
    def __init__(self, id, username, is_admin):
        self.id = id
        self.username = username
        self.is_admin = is_admin

# Hardcoded users (in a real app, use a database)
users = {
    'user1': {'id': 1, 'username': 'user1', 'password': 'password1', 'is_admin': False},
    'user2': {'id': 2, 'username': 'user2', 'password': 'password2', 'is_admin': False},
    'admin': {'id': 3, 'username': 'admin', 'password': 'adminpass', 'is_admin': True},
}

@login_manager.user_loader
def load_user(user_id):
    for username, user_info in users.items():
        if user_info['id'] == int(user_id):
            return User(user_info['id'], user_info['username'], user_info['is_admin'])
    return None

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        # Check user credentials
        if username in users and users[username]['password'] == password:
            user_info = users[username]
            user = User(user_info['id'], user_info['username'], user_info['is_admin'])
            login_user(user)
            return redirect(url_for('chat'))
        else:
            return 'Invalid credentials', 401
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/chat')
@login_required
def chat():
    is_admin = current_user.is_admin
    return render_template('index.html', is_admin=is_admin)

def generate_embeddings_from_csv(file_path, model, db_path):
    df = pd.read_csv(file_path)
    
    # Debug: Print the columns of the uploaded CSV
    print(f"Uploaded CSV columns: {df.columns.tolist()}")

    if 'input' not in df.columns or 'output' not in df.columns:
        raise ValueError("CSV must contain 'input' and 'output' columns.")
    
    input_embeddings = model.encode(df['input'].tolist())
    output_texts = df['output'].tolist()
    
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            input TEXT,
            embedding BLOB,
            output TEXT
        )
    ''')
    
    for input_text, embedding, output_text in zip(df['input'], input_embeddings, output_texts):
        c.execute("INSERT INTO embeddings (input, embedding, output) VALUES (?, ?, ?)", 
                  (input_text, embedding.tobytes(), output_text))
    
    conn.commit()
    conn.close()

def retrieve_results_from_db(query, top_k=1):
    query_embedding = model.encode([query])[0]
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT input, embedding, output FROM embeddings')
    rows = cursor.fetchall()
    results = []
    for row in rows:
        input_text, embedding_blob, output_text = row
        embedding = np.frombuffer(embedding_blob, dtype=np.float32)
        similarity = np.dot(query_embedding, embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(embedding))
        results.append((similarity, output_text))
    results.sort(key=lambda x: x[0], reverse=True)
    output_results = [result[1] for result in results[:top_k]]
    conn.close()
    return output_results

@app.route('/')
def home():
    return redirect(url_for('login'))

@app.route('/query', methods=['POST'])
@login_required
def query():
    data = request.json
    query = data['query']
    results = retrieve_results_from_db(query)
    return jsonify(results)

@app.route('/upload_csv', methods=['POST'])
@login_required
def upload_csv():
    if not current_user.is_admin:
        return jsonify({'message': 'Unauthorized'}), 403

    if 'file' not in request.files:
        return jsonify({'message': 'No file part'}), 400

    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'message': 'No selected file'}), 400

    if file:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)
        
        try:
            generate_embeddings_from_csv(file_path, model, DB_PATH)
            return jsonify({'message': 'File processed successfully'}), 200
        except Exception as e:
            return jsonify({'message': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
