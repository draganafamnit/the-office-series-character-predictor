# data_exploration.py
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from collections import Counter
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Create "results" folder if it does not exit
os.makedirs("results", exist_ok=True)

# Set seaborn style for better visuals
sns.set(style="whitegrid")

# Set input and output paths
input_path = 'data/the-office-lines.csv'
output_path = 'results/exploration_results.csv'

# Load the dataset
data = pd.read_csv(input_path)

# Basic dataset overview
print("Dataset Info:")
print(data.info())
print("\nFirst 5 rows:")
print(data.head())

# Count unique characters and their line frequencies
character_counts = data['Character'].value_counts()
total_lines = character_counts.sum()
character_percentages = (character_counts / total_lines) * 100

# Filter characters with at least 1.6% of lines
min_percentage = 1.6
top_chars = character_percentages[character_percentages >= min_percentage].head(13).index
print(f"\nTop 13 characters with at least {min_percentage}% of lines:")
print(character_percentages[top_chars].round(2))

# Display textual summary
print("\nNumber of unique characters:", len(character_counts))
print("\nCharacter line distribution:")
print(character_counts)
print("\nPercentage of total lines per character:")
print(character_percentages.round(2))

# Sentiment analysis
def get_sentiment(text):
    return TextBlob(str(text)).sentiment.polarity

data['sentiment'] = data['Line'].apply(get_sentiment)
sentiment_by_character = data.groupby('Character')['sentiment'].mean().sort_values()

# Word count per line
data['word_count'] = data['Line'].apply(lambda x: len(str(x).split()))
word_count_stats = data.groupby('Character')['word_count'].agg(['mean', 'std', 'min', 'max'])

# Top words per character (using TF-IDF)
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

data['cleaned_line'] = data['Line'].apply(clean_text)
tfidf = TfidfVectorizer(max_features=100, stop_words='english')
tfidf_matrix = tfidf.fit_transform(data['cleaned_line'])
feature_names = tfidf.get_feature_names_out()

# Get top 5 words for top characters
for char in top_chars:
    char_lines = data[data['Character'] == char]['cleaned_line']
    if len(char_lines) > 0:
        char_tfidf = tfidf.transform(char_lines)
        mean_tfidf = char_tfidf.mean(axis=0).A1
        top_words_idx = mean_tfidf.argsort()[-5:][::-1]
        top_words = [feature_names[i] for i in top_words_idx]
        print(f"\nTop 5 words for {char}: {top_words}")

# Visualizations
plt.figure(figsize=(15, 20))

# Bar plot for character distribution
plt.subplot(3, 2, 1)
character_counts[top_chars].plot(kind='bar')
plt.title('Number of Lines for Top 13 Characters')
plt.xlabel('Character')
plt.ylabel('Number of Lines')
plt.xticks(rotation=45)

# Pie chart for percentage distribution
plt.subplot(3, 2, 2)
character_percentages[top_chars].plot(kind='pie', autopct='%1.1f%%', startangle=90)
plt.title('Percentage of Lines for Top 13 Characters')
plt.ylabel('')

# Sentiment boxplot by character
plt.subplot(3, 2, 3)
sns.boxplot(x='Character', y='sentiment', data=data[data['Character'].isin(top_chars)])
plt.title('Sentiment Distribution for Top 13 Characters')
plt.xlabel('Character')
plt.ylabel('Sentiment Polarity')
plt.xticks(rotation=45)

# Sentiment histogram
plt.subplot(3, 2, 4)
sns.histplot(data['sentiment'], bins=30, kde=True)
plt.title('Overall Sentiment Distribution')
plt.xlabel('Sentiment Polarity')
plt.ylabel('Frequency')

# Word count histogram
plt.subplot(3, 2, 5)
sns.histplot(data['word_count'], bins=30, kde=True)
plt.title('Line Length Distribution (Word Count)')
plt.xlabel('Word Count')
plt.ylabel('Frequency')

# Average word count by character
plt.subplot(3, 2, 6)
word_count_stats.loc[top_chars, 'mean'].plot(kind='bar')
plt.title('Average Word Count per Line for Top 13 Characters')
plt.xlabel('Character')
plt.ylabel('Average Word Count')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# Display additional summaries
print("\nWord Count Statistics by Character")
print(word_count_stats.loc[top_chars].round(2))
print("\nAverage Sentiment by Character")
print(sentiment_by_character[top_chars].round(2))

# Save results to CSV
data[['Character', 'Line', 'sentiment', 'word_count']].to_csv(
    'results/exploration_results.csv', index=False
)
print(f"\nExploration results saved to {output_path}")
