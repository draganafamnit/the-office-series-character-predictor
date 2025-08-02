# preprocess.py
import os
import pandas as pd
import joblib
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import TfidfVectorizer

# Define input and output paths
input_path = 'data/the-office-lines.csv'
output_folder = 'preprocessed'
os.makedirs(output_folder, exist_ok=True)

# Load the dataset
data = pd.read_csv(input_path)

# Filter characters with at least 1.6% of total lines
total_lines = len(data)
character_counts = data['Character'].value_counts()
threshold = 0.016 
selected_chars = character_counts[character_counts / total_lines >= threshold].index
data = data[data['Character'].isin(selected_chars)]

# Clean text: lowercase and remove punctuation
data['Line'] = data['Line'].str.lower().str.replace(r'[^\w\s]', '')

# Apply TF-IDF vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X = vectorizer.fit_transform(data['Line'])
y = data['Character']

# Save preprocessed features and labels
save_npz(os.path.join(output_folder, 'X_preprocessed.npz'), X)
y.to_csv(os.path.join(output_folder, 'Y_preprocessed.csv.gz'), index=False, compression='gzip')
joblib.dump(vectorizer, os.path.join(output_folder, 'vectorizer.pkl'))


print("Preprocessing complete. Files saved in 'preprocessed/' folder.")
