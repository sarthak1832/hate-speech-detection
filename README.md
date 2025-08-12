# Hate Speech Detection Web App

A Flask-based web application that detects hate speech and offensive language in text using machine learning.

---

## Overview

This project uses a Logistic Regression model with TF-IDF vectorization to classify tweets into three categories: Hate Speech, Offensive Language, or Neither. The model is trained on a labeled Twitter dataset from [Kaggle](https://www.kaggle.com/datasets/mrmorj/hate-speech-and-offensive-language-dataset).

The app features:

- Real-time text input with classification
- Responsive and modern UI built with Bootstrap
- Balanced class weighting to address data imbalance
- Detailed evaluation using precision, recall, and F1-score

---

## Project Structure

- `train_model.py`: Script to train the model and save artifacts.
- `app.py`: Flask web server for user interaction and inference.
- `templates/index.html`: Frontend HTML template.
- `data/labeled_data.csv`: Dataset file (download manually).
- `model.pkl` and `vectorizer.pkl`: Saved model and vectorizer files (generated after training).
- `requirements.txt`: Project dependencies.

---

## Setup Instructions

1. Clone the repo:
    ```bash
    git clone https://github.com/yourusername/hate-speech-detection.git
    cd hate-speech-detection
    ```

2. Create and activate a virtual environment:
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # Mac/Linux
    venv\Scripts\activate     # Windows
    ```

3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/mrmorj/hate-speech-and-offensive-language-dataset) and place `labeled_data.csv` inside the `data/` folder.

5. Train the model:
    ```bash
    python train_model.py
    ```

6. Run the web app:
    ```bash
    python app.py
    ```

7. Open your browser at [http://127.0.0.1:5000](http://127.0.0.1:5000).

---

## Future Improvements

- Upgrade to transformer-based models like BERT for better contextual understanding.
- Add real-time monitoring of social media data.
- Enhance UI with frontend frameworks like React.
- We can Make an extension in Future So while scrolling through instagram or other social app/website  we can click on any picture and then with out extension option we can find it hate or not. 

---

## Technologies Used

Python, Flask, scikit-learn, pandas, Bootstrap

---

## License

This project is for educational purposes. Please cite the Kaggle dataset if used.