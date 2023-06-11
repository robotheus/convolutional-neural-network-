# a pasta kaggleDataset foi retirada de https://www.kaggle.com/c/dogs-vs-cats
# aqui realizamos o pré-processamento criando as pastas train, test e validation utilizada pelo TensorFlow

import os
import glob
import shutil
import random

def preProcessamento():
    diretorio = os.path.abspath(os.getcwd())
    
    pastaTrain = diretorio + "/dataset/train"
    pastaValidation = diretorio + "/dataset/validation"
    pastaTest = diretorio + "/dataset/teste"

    os.mkdir(diretorio + "/dataset")

    os.mkdir(pastaTrain)
    os.mkdir(pastaValidation)
    os.mkdir(pastaTest)

    os.mkdir(pastaTrain + "/dogs")
    os.mkdir(pastaTrain + "/cats")
    os.mkdir(pastaValidation + "/dogs")
    os.mkdir(pastaValidation + "/cats")
    os.mkdir(pastaTest + "/dogs")
    os.mkdir(pastaTest + "/cats")

    validacao = 0.10 
    teste = 0.20 
    
    dogTrain = glob.glob(diretorio + '/kaggleDataset/train/dog.*')
    catTrain = glob.glob(diretorio + '/kaggleDataset/train/cat.*')

    for x in dogTrain:
        valorAleatorio = random.random()
        nomeImagem = x.split("/")[-1]
        
        if valorAleatorio <= validacao:
            shutil.move(x, pastaValidation + "/dogs/" + nomeImagem)
        elif valorAleatorio > validacao and valorAleatorio <= validacao + teste:
            shutil.move(x, pastaTest + "/dogs/" + nomeImagem)
        else:
            shutil.move(x,  pastaTrain + "/dogs/" + nomeImagem)
    
    for x in catTrain:
        valorAleatorio = random.random()
        nomeImagem = x.split("/")[-1]
        
        if valorAleatorio <= validacao:
            shutil.move(x, pastaValidation + "/cats/" + nomeImagem)
        elif valorAleatorio > validacao and valorAleatorio <= validacao + teste:
            shutil.move(x, pastaTest + "/cats/" + nomeImagem)
        else:
            shutil.move(x,  pastaTrain + "/cats/" + nomeImagem)

if __name__ == '__main__':
    preProcessamento()