##Machine learning model deployment with flask

This repository contains the implementation of REST API used to generate a prediction from an ML model. The ensemble model is trained to predict th "representativeness"
of given data records. To train the model, available data is split into L parts and on each a separate model is fitted. The final model is an ensemble.

All required packages are in "requirements.txt" file.

The jupyter notebook "model_selection.ipynb" contains a comparison between different base models
and optimization of the algorithms input parameters K and L.

The chosen model is then trained in the "model_building.py" file. Trained ensemble models
are saved as well as the testing data.

To start the API execute "main.py" and open 'localhost:5000' via browser. To generate model predictions upload 'test_data.csv' file.
