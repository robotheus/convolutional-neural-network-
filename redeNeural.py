import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow import keras
from tensorflow.keras.layers import Rescaling
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, RandomFlip, RandomRotation, RandomZoom
from tensorflow.keras.callbacks import ModelCheckpoint

def RedeNeural():
    diretorio = os.path.abspath(os.getcwd())
    pastaTrain = diretorio + "/dataset/train"
    pastaValidation = diretorio + "/dataset/validation"
    pastaTest = diretorio + "/dataset/test"

    dadosTrain = image_dataset_from_directory(pastaTrain, image_size = (180,180), batch_size = 32)
    dadosValidation = image_dataset_from_directory(pastaValidation, image_size = (180,180), batch_size = 32)
    dadosTest = image_dataset_from_directory(pastaTest, image_size=(180,180), batch_size=32)

    modeloRedeNeural = keras.Sequential([Rescaling(scale = 1.0/255), 
                                         RandomFlip("horizontal"), 
                                         RandomRotation(0.1), 
                                         RandomZoom(0.2)])

    modeloRedeNeural.add(Conv2D(32, kernel_size = (3, 3), activation = 'relu'))
    modeloRedeNeural.add(BatchNormalization())
    modeloRedeNeural.add(MaxPooling2D(pool_size = (2, 2)))

    modeloRedeNeural.add(Conv2D(64, kernel_size = (3, 3), activation = 'relu'))
    modeloRedeNeural.add(BatchNormalization())
    modeloRedeNeural.add(MaxPooling2D(pool_size = (2, 2)))
    
    modeloRedeNeural.add(Conv2D(128, kernel_size = (3, 3), activation = 'relu'))
    modeloRedeNeural.add(BatchNormalization())
    modeloRedeNeural.add(MaxPooling2D(pool_size = (2, 2)))
    
    modeloRedeNeural.add(Conv2D(256, kernel_size = (3, 3), activation = 'relu'))
    modeloRedeNeural.add(BatchNormalization())
    modeloRedeNeural.add(MaxPooling2D(pool_size = (2, 2)))
    
    modeloRedeNeural.add(Flatten()) 
    modeloRedeNeural.add(Dense(256, activation = 'relu'))
    modeloRedeNeural.add(Dropout(0.5))
    modeloRedeNeural.add(Dense(1, activation = "sigmoid"))

    modeloRedeNeural.compile(loss = "binary_crossentropy", optimizer = "adam", metrics = ["accuracy"])

    callbacks = [ModelCheckpoint(filepath = "modelo1.keras", save_best_only = True, monitor = "val_loss")]
    treinamento = modeloRedeNeural.fit(dadosTrain, 
                                       epochs = 30, 
                                       validation_data = dadosValidation, 
                                       callbacks = callbacks)

    acuracia = treinamento.history["accuracy"]
    valorAcuracia = treinamento.history["val_accuracy"]
    erro = treinamento.history["loss"]
    valorErro = treinamento.history["val_loss"]
    epochs = range(1, len(acuracia) + 1)
    plt.plot(epochs, acuracia, "r", label = "acuracia no treino")
    plt.plot(epochs, valorAcuracia, "b", label = "acuracia na validacao")
    plt.xlabel("Épocas")
    plt.ylabel("%s")
    plt.title("Acurácia no Treino e Validação")
    plt.legend()
    plt.figure()
    plt.plot(epochs, erro, "r", label = "erro no treino")
    plt.plot(epochs, valorErro, "b", label = "erro na validacao")
    plt.xlabel("Épocas")
    plt.ylabel("%s")
    plt.title("Erro no Treino e Validação")
    plt.legend()
    plt.show()

    modeloFinal = keras.models.load_model("modelo1.keras")
    lixo, acuraciaTeste = modeloFinal.evaluate(dadosTest)
    porcentagemAcuracia = acuraciaTeste*100
    
    print(f'Acuracia nos testes: {porcentagemAcuracia:.2f}%')
    
if __name__ == '__main__':
    RedeNeural()