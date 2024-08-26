from sentence_transformers import SentenceTransformer
import os

# Directory where the model will be saved
model_save_path = 'models/all-MiniLM-L6-v2'

# Load the model and save it locally
model = SentenceTransformer('all-MiniLM-L6-v2')
model.save(model_save_path)
