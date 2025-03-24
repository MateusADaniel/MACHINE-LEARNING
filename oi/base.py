import os
import math
import random
import sys
import fnmatch
from pathlib import PurePath
import numpy as np
import tensorflow as tf
import visualization


# This function generates an array of arrays (a matrix?) which contains the following:
# [0] = an array of image path(s) from the specified path and respective university
# [1] = an array of label(s) corresponding to each image
#
# The way we do so is by walking through the directory tree and trying to match the proper
# parameters specified in the function. Mind: we do a reverse match, that is, we match
# from the lowest item in the tree and then work our way up to make sure we are adding the
# proper file and label to the array. This could be better worked upon as to not iterate
# through useless paths as well as include some sort of multi threading.
#
# Here's some ASCII art of how we do so:
#
# pklot=[]
# matches = []
# requirements = ['PKLotSegmented', 'PUC']
#
# 2012-09-21_06_10_10#001.jpg (lowest item in the tree, and is a match to our jpg regex)
# |
# |                                       pklot=[]
# |                                       matches=['2012-09-21_06_10_10#001.jpg']
# |
# |
# |___
#      Empty
#      |
#      |_______
#              2012-09-21
#              |
#              |____________
#                           Rainy
#                           |
#                           |________
#                                    PUC (match, requirements[1])
#                                    |
#                                    |____________
#                                                 PKLotSegmented(match, requirements[0])
#
#                                                 pklot=[['PUC','Rainy','2012-09-21', \
#                                                         'Empty', \
#                                                         'PATHTOTHEJPG']]
#
#
def pathgen(path: str, university: str, logger: object) -> [[]]:

    logger.info("Argumentos enviados para o gerador do dicionário de arquivos:")
    logger.info(f'path: {path}')
    logger.info(f'university: {university}')
    logger.info(f'logger: {logger}')

    logger.info("Iniciando geração do dicionário de arquivos")

    requirements = ['PKLotSegmented']
    pklot = {}

    if university is not None:
        requirements.append(university)

    for root, _, files in os.walk(top=path, followlinks=True):

        root_path = PurePath(root).parts

        if len(files) > 0:
            if all(requirement in root_path for requirement in requirements):
                for file in files:
                    if fnmatch.fnmatch(file, '*.jpg'):

                        university = root_path[-4]
                        date = root_path[-2]
                        state = root_path[-1]
                        file_path = os.path.abspath(os.path.join(root, file))

                        if university in pklot:
                            if date in pklot[university]:
                                if state in pklot[university][date]:
                                    pklot[university][date][state].append(
                                        file_path)
                                else:
                                    pklot[university][date][state] = [
                                        file_path]
                            else:
                                pklot[university][date] = {state: [file_path]}
                        else:
                            pklot |= {university: {date: {state: [file_path]}}}

    logger.info("Concluida a geração do dicionário de arquivos")

    return pklot


def dataset_gen(datasetpath: str, percentagetrain: float, percentagetest: float, percentagevalidation: float, university: str, overlap_days: bool, logger: object) -> [[]]:

    logger.info("Iniciando geração do dicionário do dataset")
    logger.info("Parametros enviados para o gerador de dicionário do dataset:")
    logger.info(f'datasetpath: {datasetpath}')
    logger.info(f'percentagetrain: {percentagetrain}')
    logger.info(f'percentagetest: {percentagetest}')
    logger.info(f'percentagevalidation: {percentagevalidation}')
    logger.info(f'university: {university}')
    logger.info(f'overlap_days: {overlap_days}')
    logger.info(f'logger: {logger}')

    dataset = pathgen(datasetpath, university, logger)
    train = []
    test = []
    validation = []


    if overlap_days:

        logger.info("Inicio do loop de geração com dias coincidentes")

        total_files = []

        for days in dataset[university]:
            for state in dataset[university][days]:
                for file in dataset[university][days][state]:
                    total_files.append([file, state])

        amount_train = int(math.floor(len(total_files)*percentagetrain))
        amount_test = int(math.floor(len(total_files)*percentagetest))
        amount_validation = int(math.ceil(len(total_files)*percentagevalidation))

        for _ in range(0, amount_train):
            random_index = random.randrange(0, len(total_files))
            train.append(total_files[random_index])
            total_files.pop(random_index)

        for _ in range(0, amount_test):
            random_index = random.randrange(0, len(total_files))
            test.append(total_files[random_index])
            total_files.pop(random_index)

        for _ in range(0, amount_validation):
            random_index = random.randrange(0, len(total_files))
            validation.append(total_files[random_index])
            total_files.pop(random_index)

        logger.info("Concluído loop de geração com dias coincidentes")

    else:

        logger.info("Inicio do loop de geração sem dias coincidentes")

        days = list(dataset[university])

        train_days = []
        test_days = []
        validation_days = []

        amount_train = int(math.floor(len(days)*percentagetrain))
        amount_test = int(math.ceil(len(days)*percentagetest))
        amount_validation = int(math.ceil(len(days)*percentagevalidation))

        for _ in range(0, amount_train):
            random_index = random.randrange(0, len(days))
            train_days.append(days[random_index])
            days.pop(random_index)

        for _ in range(0, amount_test):
            random_index = random.randrange(0, len(days))
            test_days.append(days[random_index])
            days.pop(random_index)

        for _ in range(0, amount_validation):
            random_index = random.randrange(0, len(days))
            validation_days.append(days[random_index])
            days.pop(random_index)

        for train_day in train_days:
            for state in dataset[university][train_day]:
                for file_path in dataset[university][train_day][state]:
                    train.append([file_path, state])

        for test_day in test_days:
            for state in dataset[university][test_day]:
                for file_path in dataset[university][test_day][state]:
                    test.append([file_path, state])

        for validation_day in validation_days:
            for state in dataset[university][validation_day]:
                for file_path in dataset[university][validation_day][state]:
                    validation.append([file_path, state])

        logger.info("Concluído loop de geração sem dias coincidentes")

    logger.info("Concluido geração do dicionário do dataset")
    logger.info(f'Quantidade de dias de teste: {len(test)}')
    logger.info(f'Quantidade de dias de treino: {len(train)}')
    logger.info(f'Quantidade de dias de validation: {len(validation)}')

    return_value = {'train': train, 'test': test, 'validation': validation}
    return return_value

# Converts the dataset labels from their string counterparts to ints.
def convert_label(label):
    match label:
        case 'Empty':
            return int(0)
        case 'Occupied':
            return int(1)
        case 0:
            return 'Empty'
        case 1:
            return 'Occupied'
        case _:
            sys.exit("Error: Invalid label conversion, input: {label}")
