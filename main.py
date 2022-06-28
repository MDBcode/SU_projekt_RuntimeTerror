from keras.models import load_model
from keras.utils import img_to_array
from keras.utils import load_img
from numpy import expand_dims
from instance_normalization import InstanceNormalization

PIX2PIX_MODEL_PATH = "models/saved_models/pix2pix_model_054800.h5"
CYCLEGAN_MODEL_PATH_SAT_TO_MAP = "models/saved_models/cycleGAN_model_AtoB_004384.h5"
CYCLEGAN_MODEL_PATH_MAP_TO_SAT = "models/saved_models/cycleGAN_model_BtoA_004384.h5"
CGAN_MODEL_PATH = "models/saved_models/..."  # dodati model

cust = {"InstanceNormalization": InstanceNormalization}

model_pix2pix = load_model(PIX2PIX_MODEL_PATH)
model_cycleGAN_sat_to_map = load_model(CYCLEGAN_MODEL_PATH_SAT_TO_MAP, cust)
model_cycleGAN_map_to_sat = load_model(CYCLEGAN_MODEL_PATH_MAP_TO_SAT, cust)
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

def predict_cycleGAN_sat_to_map(filename):
    img = load_image(filename)
    predicted_map = model_cycleGAN_sat_to_map.predict(img)
    predicted_map = (predicted_map + 1) / 2.0
    return predicted_map

def predict_cycleGAN_map_to_sat(filename):
    img = load_image(filename)
    predicted_satellite = model_cycleGAN_map_to_sat.predict(img)
    predicted_satellite = (predicted_satellite + 1) / 2.0
    return predicted_satellite
