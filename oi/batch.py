#import os
#import time
#import tensorflow as tf
#import base
#import threecvnn
#import mobilenet
#import efficientnet
#import resnet50
#import visualization
#import copy
#from pathlib import PurePath
#import pandas as pd
#
#def batch_data(path, train, test, validation, universities):
#
#    if path == None:
#        path = './'
#
#    if train == None:
#        train = 0.4
#
#    if test == None:
#        test = 0.5
#
#    if validation == None:
#        validation = 0.1
#
#    if universities == None:
#        universities = ['PUC', 'UFPR04', 'UFPR05']
#
#
#    models = [threecvnn.model,
#              mobilenet.static,
#              mobilenet.dynamic,
#              efficientnet.static,
#              efficientnet.dynamic,
#              resnet50.static,
#              resnet50.dynamic]
#
#
#    for model in models:
#
#        if model().name == 'threecvnn':
#            img_size = 32
#        else:
#            img_size = 128
#
#        for university in universities:
#
#            dataset = base.dataset_gen(path,train,test,validation,university, False)
#
#            train_dataset, test_dataset, validation_dataset = dataset
#
#            model_path = f'./tests/{model().name}/{university}'
#
#            model_run = model()
#
#            university = university
#
#            os.makedirs(name=f'{model_path}/ending/',
#                        exist_ok=True)
#
#            checkpoint_filepath = f'{model_path}/best.keras'
#
#            log_file = open(f'{model_path}/log.txt', 'w')
#            log_file.write(f'Modelo: {model().name}\nUniversidade: {university}\nArquivos de treino:\n{train_dataset}\nArquivos de teste:\n{test_dataset}\nArquivos de validacao:\n {validation_dataset}\n')
#
#
#            base.train_model(model_run,train_dataset,validation_dataset,test_dataset, checkpoint_filepath, img_size, model_path)
#
#            model_run.save(f'{model_path}/ending/model.keras')
#
#            accuracy_file = open(f'{model_path}/ending/accuracy.txt', 'w')
#            accuracy_file.write(f'Accuracy: {model_run.get_metrics_result()}')
#
#            model_run = tf.keras.models.load_model(f'{model_path}/best.keras')
#
#            accuracy_file = open(f'{model_path}/ending/accuracy.txt', 'w')
#            accuracy_file.write(f'Accuracy: {model_run.get_metrics_result()}')
#
#
#def gen_graphs(paths):
#    for path in paths:
#        rootpath = PurePath(path).parts
#        savepath = os.path.join(rootpath,'graphs')
#        os.makedirs(name=savepath,exist_ok=True)
#        log_data = pd.read_csv(f'{path}/training.log', sep=',', engine='python')
#        visualization.gen_graphs(log_data,savepath)
#