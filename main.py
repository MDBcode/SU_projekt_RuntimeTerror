from keras.models import load_model
from keras.utils import img_to_array
from keras.utils import load_img
from numpy import expand_dims

PIX2PIX_MODEL_PATH = "models/model_054800.h5"
CGAN_MODEL_PATH = "models/"  # dodati model

model_pix2pix = load_model(PIX2PIX_MODEL_PATH)
# model_cgan = load_model(CGAN_MODEL_PATH)

def load_image(filename):
    pixels = load_img(filename)
    pixels = img_to_array(pixels)
    pixels = (pixels - 127.5) / 127.5
    pixels = expand_dims(pixels, 0)
    return pixels


def predict_cgan(input_img):
    pass


def predict_pix2pix(filename):
    img = load_image(filename)
    gen_image = model_pix2pix.predict(img)
    gen_image = (gen_image + 1) / 2.0
    return gen_image[0]
