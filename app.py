from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        text = request.form['text']
        text_vectorized = vectorizer.transform([text])
        prediction = model.predict(text_vectorized)[0]

        if prediction == 0:
            result = "Hate Speech Detected"
        elif prediction == 1:
            result = "Offensive Language Detected"
        else:
            result = "No Hate or Offensive Language Detected"

    return render_template('index.html', result=result)

if __name__ == "__main__":
    app.run(debug=True)