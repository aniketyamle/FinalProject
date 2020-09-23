from flask import Flask, render_template, request, flash, redirect, url_for
from werkzeug.utils import secure_filename
import os

################ new ###########################
from keras.models import load_model
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.preprocessing.image import load_img, img_to_array
from skimage.transform import resize

import matplotlib.pyplot as plt

# Define IoU metric


def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)


model = load_model('model-tgs-salt-1.h5',custom_objects={'mean_iou': mean_iou})

# model._make_predict_function()
################ new ###########################


UPLOAD_FOLDER = 'static/input'
OUTPUT_FOLDER = 'static/output'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER



@app.route('/', methods=['GET', 'POST'])
def upload():

    if request.method == 'GET':
        return render_template('home.html')
    
    if request.method == 'POST':
        file = request.files.get('inputImage', '')
        filename = secure_filename(file.filename)
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(path)

        ################ new ###########################
        img = load_img(path)

        x = img_to_array(img)[:, :, 1]
        x = resize(x, (128, 128, 1), mode='constant', preserve_range=True)
        X_train = np.zeros((1, 128, 128, 1), dtype=np.uint8)
        X_train[0] = x

        preds_test = model.predict(X_train, verbose=1)
        preds_train_t = (preds_test > 0.5).astype(np.uint8)

        tmp = np.squeeze(preds_train_t[0]).astype(np.float32)
        plt.imshow(np.dstack((tmp, tmp, tmp)))

        output = os.path.join(app.config['OUTPUT_FOLDER'], filename)
        plt.savefig(output)
        return render_template('home.html', input=path, output=output) 


if __name__ == '__main__':
    app.run(debug=True)
