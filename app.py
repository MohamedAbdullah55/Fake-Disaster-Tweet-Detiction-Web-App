'''
Author Name : Mohamed Abdullah
Author link : https://github.com/MohamedAbdullahKamel
                  https://www.kaggle.com/mohamedabdullah
                  https://www.linkedin.com/in/mohamedabdullahkamel/
Version : v.1.0.0
Organization : brainless

'''    
from flask import Flask
from flask import render_template
from flask import request
from flask import jsonify
from flask import redirect
from flask import url_for
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import numpy as np
from sklearn.externals import joblib
nltk.download('stopwords')

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html" )

@app.route("/api", methods=["GET","POST"])
def api():
    if request.method == "POST":
        words = joblib.load('words.pkl')
        model = joblib.load('model.pkl')
        pstem = PorterStemmer()

        tweet = request.form["tweet"]
        text = tweet
        text = re.sub("[^a-zA-Z]", ' ', text)
        text = text.lower()
        text = text.split()
        text = [pstem.stem(word) for word in text if not word in set(stopwords.words('english'))]
        text = ' '.join(text)

        query = []
        for word in words:
            if word in text:
                query.append(1)
            else:
                query.append(0)

        prediction = list(model.predict(np.matrix(query)))[0]

        if prediction == 1:
            msg = "Approximately 70%, your tweet Not Fake (Real tweet)."
            return render_template("index.html", msg=msg, tweet=tweet)
        else:
            error = "Approximately 70%, your tweet Fake"
            return render_template("index.html", error=error, tweet=tweet)
    else:
        return redirect(url_for("index"))

if __name__ == '__main__':
    app.run(debug=True)
