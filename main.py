from flask import Flask, request, render_template, session
import pandas as pd
import os
import pickle
import numpy as np
from sklearn.metrics import mean_squared_error

# loading the ensemble models
ensemble = []
for filename in os.listdir(os.path.join(os.getcwd(), 'ensemble')):
    ensemble.append(pickle.load(open(os.path.join(os.getcwd(),'ensemble\\',filename), 'rb')))

UPLOAD_FOLDER = os.path.join('staticFiles', 'uploads')
ALLOWED_EXTENSIONS = {'csv'}

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'secretkey'

#loading the testing data throuh a route
@app.route('/', methods=['GET', 'POST'])
def uploadFile():
    if request.method == 'POST':
        # upload file flask
        f = request.files.get('file')

        # Extracting uploaded file name
        data_filename = f.filename

        f.save(os.path.join(app.config['UPLOAD_FOLDER'],
                            data_filename))

        session['uploaded_data_file_path'] = os.path.join(app.config['UPLOAD_FOLDER'], data_filename)

        return render_template('index2.html')
    return render_template("index.html")


@app.route('/result')
def result():
    # Uploaded File Path
    data_file_path = session.get('uploaded_data_file_path', None)
    # read csv test file
    test_df = pd.read_csv(data_file_path, encoding='unicode_escape')

    predictions = []
    for model in ensemble:
        x_test = test_df.iloc[:,:-1]
        predictions.append(list(model.predict(x_test)))

    final_prediction = np.mean(np.array(predictions), axis=0)

    mse = mean_squared_error(test_df.iloc[:,-1], final_prediction)

    test_df['predicted_representativity'] = final_prediction


    return render_template('result.html', prediction=test_df.to_html(), mse=mse)


if __name__ == "__main__" :
    app.run(debug=True)