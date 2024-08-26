import pandas as pd
import sqlite3
import numpy as np
from sentence_transformers import SentenceTransformer

def generate_embeddings_from_csv(csv_file_path, model_path, db_path):
    # Load pre-trained model from the local directory
    model = SentenceTransformer(model_path)

    # Load CSV file
    df = pd.read_csv(csv_file_path)

    # Generate embeddings for the input column
    input_embeddings = model.encode(df['input'].tolist())

    # Connect to SQLite database (or create it if it doesn't exist)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create table if it doesn't exist
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS embeddings (
        id INTEGER PRIMARY KEY,
        input TEXT,
        embedding BLOB,
        output TEXT
    )
    ''')

    # Insert embeddings into the database
    for i, (input_text, embedding, output) in enumerate(zip(df['input'], input_embeddings, df['output'])):
        cursor.execute('''
        INSERT INTO embeddings (id, input, embedding, output) VALUES (?, ?, ?, ?)
        ''', (i, input_text, embedding.tobytes(), output))

    # Commit and close the connection
    conn.commit()
    conn.close()

if __name__ == "__main__":
    csv_file_path = 'data/main.csv'  # Update with the path to your CSV file
    model_path = 'models/all-MiniLM-L6-v2'  # Path to the local model directory
    db_path = 'embeddings.db'  # Path to SQLite database file

    # Generate embeddings and store them in the SQLite database
    generate_embeddings_from_csv(csv_file_path, model_path, db_path)
