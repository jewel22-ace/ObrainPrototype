import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

def generate_embeddings_from_csv(csv_file_path, model_name, persist_directory, collection_name="embeddings_collection"):
    # Load pre-trained model
    model = SentenceTransformer(model_name)

    # Initialize Chroma DB with persistence settings
    client = chromadb.Client(Settings(persist_directory=persist_directory))

    # Check if the collection exists
    try:
        collection = client.get_collection(name=collection_name)
    except Exception as e:
        print(f"Collection not found, creating new collection: {e}")
        # Create a collection in Chroma DB if it does not exist
        collection = client.create_collection(name=collection_name)

    # Load CSV file
    df = pd.read_csv(csv_file_path)

    # Generate embeddings for the input column
    input_embeddings = model.encode(df['input'].tolist())

    # Add embeddings, inputs, and outputs to the collection
    documents = df['input'].tolist()
    metadatas = [{'id': str(i), 'output': output} for i, output in enumerate(df['output'].tolist())]
    ids = [str(i) for i in range(len(documents))]

    # Convert numpy array to list
    input_embeddings = [embedding.tolist() for embedding in input_embeddings]

    collection.add(documents=documents, embeddings=input_embeddings, metadatas=metadatas, ids=ids)

def retrieve_results_from_chroma(query, model_name, persist_directory, collection_name="embeddings_collection", top_k=1):
    # Load pre-trained model
    model = SentenceTransformer(model_name)

    # Generate embedding for the query
    query_embedding = model.encode([query])[0]

    # Initialize Chroma DB with persistence settings
    client = chromadb.Client(Settings(persist_directory=persist_directory))

    # Retrieve the collection
    collection = client.get_collection(name=collection_name)

    # Retrieve results by querying Chroma DB collection
    results = collection.query(query_embeddings=[query_embedding.tolist()], n_results=top_k)

    # Extract and return the relevant output rows
    output_results = [res['output'] for res in results['metadatas'][0]]

    return output_results

def continuous_query(model_name, persist_directory, collection_name="embeddings_collection"):
    while True:
        query = input("Enter your query (or type 'exit' to stop): ")
        if query.lower() == 'exit':
            break
        results = retrieve_results_from_chroma(query, model_name, persist_directory, collection_name)
        print("Search results:")
        for i, result in enumerate(results, 1):
            print(f"{i}. {result}")

# Example usage for generating embeddings
csv_file_path = 'data/main.csv'  # Update with the path to your CSV file
model_name = 'all-MiniLM-L6-v2'  # Model name for SentenceTransformer
persist_directory = 'chroma_db'  # Update with your desired directory
collection_name = 'embeddings_collection'  # Optional: Provide a custom name for the collection

# Only run this once to generate and store embeddings
generate_embeddings_from_csv(csv_file_path, model_name, persist_directory, collection_name)

# Run continuous query function
continuous_query(model_name, persist_directory, collection_name)
