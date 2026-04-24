from flask import Flask, render_template, request
import numpy as np
import pandas as pd

import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn import neighbors

app = Flask(__name__)

df = pd.read_csv('df.csv')


def segregation(data):
    values = []
    for val in data.average_rating:
        if 0 <= val <= 1:
            values.append("Between 0 and 1")
        elif 1 < val <= 2:
            values.append("Between 1 and 2")
        elif 2 < val <= 3:
            values.append("Between 2 and 3")
        elif 3 < val <= 4:
            values.append("Between 3 and 4")
        elif 4 < val <= 5:
            values.append("Between 4 and 5")
        else:
            values.append("NaN")
    return values


df['Ratings_Dist'] = segregation(df)

books_features = pd.concat([df['Ratings_Dist'].str.get_dummies(sep=","), df['average_rating'], df['ratings_count']],
                           axis=1)

min_max_scaler = MinMaxScaler()
books_features = min_max_scaler.fit_transform(books_features)

model = neighbors.NearestNeighbors(n_neighbors=6, algorithm='ball_tree')
model.fit(books_features)
distance, indices = model.kneighbors(books_features)


def get_index_from_name(name):
    return df[df["title"] == name].index.tolist()[0]


def print_similar_books(query=None, id=None):
    recommendations = []
    if id:
        for book_id in indices[id][1:]:
            recommendations.append(df.iloc[book_id]["title"])
    if query:
        found_id = get_index_from_name(query)
        for book_id in indices[found_id][1:]:
            recommendations.append(df.iloc[book_id]["title"])
    return recommendations


@app.route('/', methods=['GET', 'POST'])
def index():    
    result = None
    if request.method == 'POST':
        name = request.form['book_name']
        result = print_similar_books(query=name)
    return render_template('index.html', result=result)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/analysis')
def analysis():
    return render_template('analysis.html')

if __name__ == '__main__':
    app.run(debug=True)
