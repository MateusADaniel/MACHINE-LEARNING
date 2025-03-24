import argparse
import logging
import base
import models
import os
import matplotlib.pyplot as plt
import json
plt.switch_backend('Agg')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset-path', '-dp', default='./', required=False,
                        type=str, help='Specifies the path in which the dataset\
                                            lives in')
    parser.add_argument('--train', '-tr', default='0.4',
                        required=False, type=float)
    parser.add_argument('--test', '-tt', default='0.5',
                        required=False, type=float)
    parser.add_argument('--validation', '-vf', default='0.1',
                        required=False, type=float)
    parser.add_argument('--university', '-uni',
                        default='UFPR04', required=False, type=str)
    parser.add_argument('--model-path', '-mp',
                        default=None, required=True, type=str)
    parser.add_argument('--gen-graphs', '-gg', default=False,
                        required=False, type=bool)
    parser.add_argument('--log', '-l', default=False,
                        required=False, type=bool)
    parser.add_argument('--log-path', '-lp', default=False,
                        required=False, type=str)
    parser.add_argument('--overlap-days', '-ol',
                        default=False, required=False, type=bool)
    parser.add_argument('--model', '-m', default=None,
                        required=True, type=str)
    parser.add_argument('--random', '-r', default=True,
                        required=False, type=bool)
    parser.add_argument('--epochs', '-e', default=5, required=False, type=int)
    parser.add_argument('--pretrained-weights', '-pw', default=None,
                        required=False, type=str)
    parser.add_argument('--dataset-json', '-js',
                        default=None, required=False, type=str)
    parser.add_argument('--dropout', '-dropout', default=0, required=False, type=str)

    dataset_path = parser.parse_args().dataset_path
    train = parser.parse_args().train
    test = parser.parse_args().test
    validation = parser.parse_args().validation
    university = parser.parse_args().university.upper()
    model_path = parser.parse_args().model_path
    gen_graphs = parser.parse_args().gen_graphs
    log = parser.parse_args().log
    overlap_days = parser.parse_args().log
    model = parser.parse_args().model.lower()
    epochs = parser.parse_args().epochs
    random = parser.parse_args().random
    log_path = parser.parse_args().log_path
    pretrained_weights = parser.parse_args().pretrained_weights
    dataset_json = parser.parse_args().dataset_json
    dropout = parser.parse_args().dropout

    os.makedirs(model_path, exist_ok=True)

    if not log_path:
        log_path = f'{model_path}/logs.txt'

    if log:
        logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p',
                            filename=log_path, encoding='utf-8', level=logging.INFO)

    logger = logging.getLogger(__name__)

    logger.info(f'Iniciando script com os seguintes parametros:')
    logger.info(f'dataset_path: {dataset_path}')
    logger.info(f'train: {train}')
    logger.info(f'test: {test}')
    logger.info(f'validation: {validation}')
    logger.info(f'university: {university}')
    logger.info(f'model_path: {model_path}')
    logger.info(f'gen_graphs: {gen_graphs}')
    logger.info(f'log: {log}')
    logger.info(f'overlap_days: {overlap_days}')
    logger.info(f'model: {model}')
    logger.info(f'epochs: {epochs}')
    logger.info(f'random: {random}')
    logger.info(f'dropout: {dropout}')
    logger.info(f'log_path: {log_path}')

    models = {'threecvnn': models.ThreeCvnn,
            'whale': models.Whale,
            'simple': models.Simple,
            'siamese': models.SiameseTraining,
            'threecvnnencoder': models.ThreeCvnn_Encoder,
            'threecvnnclassifier': models.ThreeCvnnClassifier
            }

    if model == 'threecvnnclassifier' and pretrained_weights != None:
        model = models[model]

        if dataset_json != None:
            dataset = json.load(open(dataset_json))
        else:
            dataset = base.dataset_gen(dataset_path, train, test, validation,
                                        university, overlap_days, logger)

        model = model(random=random, dataset=dataset,
                    model_path=model_path, epochs=epochs, logger=logger,
                    pretrained_weights=pretrained_weights)

    elif model == 'threecvnnclassifier' and pretrained_weights == None:
        os.exit('Missing pretrained weights for model threecvnnclassifier')

    else:
        model = models[model]

        if dataset_json != None:
            dataset = json.load(open(dataset_json))
        else:
            dataset = base.dataset_gen(dataset_path, train, test, validation,
                                        university, overlap_days, logger)

        model = model(random=random, dataset=dataset,
                    model_path=model_path, epochs=epochs, dropout=dropout, logger=logger)

    model.visualization_pairs()


    model.train()

    model.visualization()

    #model.visualization_best3()

    model.visualization_best()

    model.visualization_best2()

    model.pairs_visualization()


