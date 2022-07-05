from contextlib import nullcontext
from distutils.log import debug
from itertools import product
from flask import Flask, redirect, url_for, render_template, request
from email import header
from operator import index
from flask import Flask, request, render_template, jsonify
from model import SentimentRecommenderModel

app = Flask(__name__)
sentiment_model = SentimentRecommenderModel()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def prediction():
    # get user from the html form
    user = request.form['userName']
    print(user)
    if user is not '':
        # convert text to lowercase
        user = user.lower()
        products = sentiment_model.getSentimentRecommendations(user)

        if(not(products is None)):
            print(f"retrieving items....{len(products)}")
            print(products)
            return render_template("index.html" , contents = products['name'],
            counts =products['pos_sentiment_percent'] ,
            user_name = user)
        else:
             return render_template('index.html')
    else:
        return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)