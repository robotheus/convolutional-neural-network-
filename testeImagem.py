import sys
import tensorflow as tf
import numpy as np
from tensorflow import keras

def TestaImagem(nomeModelo, nomeImagem):
    modeloRN = keras.models.load_model(nomeModelo)
    
    imagem = tf.keras.utils.load_img(nomeImagem, target_size = (180, 180))
    imgArray = tf.keras.utils.img_to_array(imagem)
    imgArray = np.expand_dims(imgArray, axis = 0)
    
    pred = (modeloRN.predict(imgArray) > 0.5).astype('int32')[0][0]

    if pred:
        print("Imagem de um dog")
    else: print("Imagem de um cat")

def main():
    nomeImagem = sys.argv[1]
    nomeModelo = sys.argv[2]
    
    TestaImagem(nomeModelo, nomeImagem)

if __name__ == '__main__':
    main()