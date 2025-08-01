# The Office - Character Predictor
A machine learning project that predicts the most likely character from "The Office" to say a given user-input sentence.

## Link to the Dataset
https://www.kaggle.com/datasets/fabriziocominetti/the-office-lines

## Project Structure
- `data/` - Input dataset (`the-office-lines.csv`)
- `requirements.txt` - Python dependencies
- `data_exploration.py` - Dataset analysis : Exploratory Data Analysis (EDA) and initial insights
- `preprocess.py` - Data preprocess : cleaning and feature extraction
- `train.py` - Main training script
- `predict.py` - Loads trained model and lets the user input a sentence


## How to Run
Create new virtual environment, clone the repo, install requirements and then run scripts
```bash
git clone https://github.com/draganafamnit/the-office-series-character-predictor.git
pip install -r requirements.txt
```
## Notes
This repo does not include output files (e.g., trained models, reports, plots), as they will be described in a separate report document.
