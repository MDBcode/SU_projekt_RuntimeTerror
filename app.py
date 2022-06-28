from flask import Flask, render_template, request
import os
import main
from PIL import Image
import io
import base64
import numpy as np

app = Flask(__name__)

path = os.getcwd()
UPLOAD_FOLDER = os.path.join(path, 'static', 'uploads')
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/', methods=['POST'])
def uploadFile():
    selected_method = request.form['model_method']
    uploaded_img = request.files['uploaded-file']
    if uploaded_img.filename != "":
        filename = uploaded_img.filename
        img_full_path = os.path.join(UPLOAD_FOLDER, filename)
        uploaded_img.save(img_full_path)
        img_rel_path = "uploads/" + filename
        if selected_method == "1":  # TODO cGAN
            result = main.predict_cgan(filename)
            return
        elif selected_method == "2":
            result = main.predict_pix2pix(os.path.join(UPLOAD_FOLDER, filename))
            formatted = (result * 255 / np.max(result)).astype('uint8')
            img_result = Image.fromarray(formatted)
            data = io.BytesIO()
            img_result.save(data, "PNG")
            encoded_img_data = base64.b64encode(data.getvalue())
            return render_template("result_pix2pix.html", map=encoded_img_data.decode('utf-8'), satellite=img_rel_path)
        elif selected_method == "3":
            result = main.predict_cycleGAN_sat_to_map(os.path.join(UPLOAD_FOLDER, filename))
            result = result.reshape((256,256,3))
            formatted = (result * 255 / np.max(result)).astype('uint8')
            img_result = Image.fromarray(formatted)
            data = io.BytesIO()
            img_result.save(data, "PNG")
            encoded_img_data = base64.b64encode(data.getvalue())
            return render_template("result_cycleGAN.html", map=encoded_img_data.decode('utf-8'), satellite=img_rel_path)
        elif selected_method == "4":
            result = main.predict_cycleGAN_map_to_sat(os.path.join(UPLOAD_FOLDER, filename))
            result = result.reshape((256,256,3))
            formatted = (result * 255 / np.max(result)).astype('uint8')
            img_result = Image.fromarray(formatted)
            data = io.BytesIO()
            img_result.save(data, "PNG")
            encoded_img_data = base64.b64encode(data.getvalue())
            return render_template("result_cycleGAN2.html", map=img_rel_path, satellite=encoded_img_data.decode('utf-8'))
    return home()


if __name__ == '__main__':
    app.run(debug=True)
