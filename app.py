import nltk
nltk.download('stopwords')
nltk.download('punkt_tab')

from flask import Flask, render_template, request
import pickle as pkl
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords

app = Flask(__name__)

@app.route('/', methods = ['GET','POST'])
def home():
    return render_template('index.html')

def transform_msg(msg):
    msg = msg.lower()
    tokens = nltk.word_tokenize(msg)

    y = []
    for i in tokens:
       if i.isalnum() and i not in stopwords.words('english'):
           y.append(i)
    stems = []
    stemmer = SnowballStemmer('english')
    for i in y:
        stem = stemmer.stem(i)
        stems.append(stem)

    return " ".join(stems)

@app.route('/predict', methods=['GET','POST'])
def predict():

    li = []
    msg = transform_msg(request.form['msg'])
    li.append(msg)
    vector = pkl.load(open('vectorizer.pkl','rb'))
    data = vector.transform(li).toarray()
    model = pkl.load(open('spam_filter.pkl','rb'))
    prediction = model.predict(data)[0]
    result = 'Spam' if prediction == 1 else 'Ham'

    return render_template('index.html', result = result, prediction = prediction)
if __name__ == '__main__':
    app.run(debug=True)
