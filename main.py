from flask import Flask, render_template, request,flash,redirect
from werkzeug.utils import secure_filename
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import sklearn
from pdfminer.high_level import extract_text
import os
import io
import pickle
# Import package
from wordcloud import WordCloud, STOPWORDS
from PIL import Image
app = Flask(__name__)

def extract_text_from_pdf(pdf_path):
    return extract_text(pdf_path)


def serve_pil_image(pil_img):

    img_io = io.BytesIO()
    pil_img.save(img_io, 'jpeg', quality=100)
    img_io.seek(0)
    img = base64.b64encode(img_io.getvalue()).decode('ascii')
    img_tag = f'<img src="data:image/jpg;base64,{img}" class="img-fluid"/>'
    return img_tag

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/uploader', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            print('No file part')
            return redirect(request.url)
        f = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if f.filename == '':
            print('No selected file')
            return 'No selected file'
        else:
            f.save(secure_filename(f.filename))
            text = extract_text_from_pdf(f.filename)
            text = os.linesep.join([s for s in text.splitlines() if s])
            text = text.replace('\r', ' ')
            text = text.replace('\n', ' ')
            prediction = pickled_model.predict(vectorizer.transform([text]))
            output=le.classes_[prediction[0]]
            print(output)
            # Import image to np.array
            mask = np.array(Image.open('upvote.png'))
            # Generate wordcloud
            wordcloud = WordCloud(width=3000, height=2000, random_state=1, background_color='white', colormap='Set2',
                                  collocations=False, stopwords=STOPWORDS, mask=mask).generate(text)
            wordcloud.to_file('{}.png'.format(f.filename[:-4]))
            #image=r'C:\Users\aakash.yadav\PycharmProjects\resumeScreening\{}.png'.format(f.filename[:-4])
        return render_template('index.html', prediction_text='Th resume is classified as {} candidate'.format(output))



if __name__ == '__main__':
    pickled_model = pickle.load(open('model.pkl', 'rb'))
    vectorizer = pickle.load(open("vectorizer.pickle", "rb"))
    le = pickle.load(open("labelEncoder.pickle", "rb"))
    app.run(debug=True)