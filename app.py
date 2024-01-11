from flask import Flask, render_template, request, redirect, url_for
from content_recommender import inputs
from collab_recommender import getRecommendations_ItemBased, getRecommendations_UserBased
from numpy import random

app = Flask(__name__)


@app.route("/")
def index():
    r = random.randint(7473)
    result = getRecommendations_UserBased(r, 5)
    return render_template('index.html', result=result, user=r)


# Route to handle form submission
@app.route('/submit', methods=['POST'])
def submit():
    # Get form data
    ingredients = request.form['ingredients'].split(',')
    min_cal = float(request.form['min'])
    max_cal = float(request.form['max'])
    tags = request.form['tags'].split(',')

    # Process the form data using the external function
    result = inputs(ingredients, min_cal, max_cal, tags)

    # Render the new page with processed data
    return render_template('result.html', result=result)


if __name__ == "__main__":
    app.run(debug=True)
