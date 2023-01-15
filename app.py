from flask import Flask, request, jsonify

from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

tfidf = TfidfVectorizer(max_features=3000)
# load the vectorizer
vectorizer = pickle.load(open('vectorizer', 'rb'))

# load the model
model = pickle.load(open('game_tweet_model.pkl', 'rb'))

import nltk
# nltk.download('stopwords')

from nltk.corpus import stopwords
# stopwords.words('english')

import string
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

app = Flask(__name__)


def transform_text(text):
    text = text.lower()  # 1. lower case
    text = nltk.word_tokenize(text)  # 2. Tokenize

    y = []
    for i in text:
        if i.isalnum():  # 3. remove special characters
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:  # 4. stopwords and punctuation
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))  # 5. Stemming

    return " ".join(y)


text = transform_text('cyka fuck do naught want two hear it ')
print(text)
# vector_input = tfidf.transform(text)
# print(vector_input)
result = model.predict(vectorizer.transform([text]))
print(result[0])


@app.route('/predict', methods=['POST'])
def predict():
    # text = input("Enter a Tweet : ")
    text = request.form.get('text')

    tweet = transform_text(text)
    result = model.predict(vectorizer.transform([tweet]))
    print(result[0])
    # results = {'tweet': transform_text(text)}

    return jsonify({'tweets': str(result[0])})


@app.route('/')
def home():
    return "Hello World"


if __name__ == '__main__':
    app.run(debug=True)
