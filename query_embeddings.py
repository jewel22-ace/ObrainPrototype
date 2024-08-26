# query_embeddings.py

import sqlite3
import numpy as np
from sentence_transformers import SentenceTransformer

def retrieve_results_from_db(query, model_name, db_path, top_k=1):
    # Load pre-trained model
    model = SentenceTransformer(model_name)

    # Generate embedding for the query
    query_embedding = model.encode([query])[0]

    # Connect to SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Retrieve all embeddings and their metadata
    cursor.execute('SELECT id, input, embedding, output FROM embeddings')
    rows = cursor.fetchall()

    # Calculate similarities
    results = []
    for row in rows:
        id, input_text, embedding_blob, output = row
        embedding = np.frombuffer(embedding_blob, dtype=np.float32)
        similarity = np.dot(query_embedding, embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(embedding))
        results.append((similarity, output))

    # Sort results by similarity
    results.sort(key=lambda x: x[0], reverse=True)

    # Extract and return the top_k results
    output_results = [result[1] for result in results[:top_k]]

    # Close the connection
    conn.close()

    return output_results

def continuous_query(model_name, db_path):
    while True:
        query = input("Enter your query (or type 'exit' to stop): ")
        if query.lower() == 'exit':
            break
        results = retrieve_results_from_db(query, model_name, db_path)
        print("Search results:")
        for i, result in enumerate(results, 1):
            print(f"{i}. {result}")

if __name__ == "__main__":
    model_name = 'all-MiniLM-L6-v2'  # Model name for SentenceTransformer
    db_path = 'embeddings.db'  # Path to SQLite database file

    # Run continuous query function
    continuous_query(model_name, db_path)
