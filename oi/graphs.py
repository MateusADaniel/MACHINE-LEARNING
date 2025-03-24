from pathlib import PurePath
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import base

import tensorflow as tf

path = './'
train = 0.4
test = 0.5
validation = 0.1

universities = ['UFPR04','UFPR05','PUC']

model_paths = ['/mnt/images/vitorhugo/test_results/model_3cvnn/UFPR04/checkpoint',
               '/mnt/images/vitorhugo/test_results/mdl_mbnetv3lg_ntp_static/UFPR04/checkpoint',
               '/mnt/images/vitorhugo/test_results/mdl_mbnetv3lg_ntp_dynamic/UFPR04/checkpoint',
               '/mnt/images/vitorhugo/test_results/mdl_efficientnet_ntp_static/UFPR04/checkpoint',
               '/mnt/images/vitorhugo/test_results/mdl_efficientnet_ntp_dynamic/UFPR04/checkpoint',
               '/mnt/images/vitorhugo/test_results/mdl_rsnt50_ntp_static/UFPR04/checkpoint',
               '/mnt/images/vitorhugo/test_results/mdl_rsnt50_ntp_dynamic/UFPR04/checkpoint',
               '/mnt/images/vitorhugo/test_results/model_3cvnn/UFPR05/checkpoint',
               '/mnt/images/vitorhugo/test_results/mdl_mbnetv3lg_ntp_static/UFPR05/checkpoint',
               '/mnt/images/vitorhugo/test_results/mdl_mbnetv3lg_ntp_dynamic/UFPR05/checkpoint',
               '/mnt/images/vitorhugo/test_results/mdl_efficientnet_ntp_static/UFPR05/checkpoint',
               '/mnt/images/vitorhugo/test_results/mdl_efficientnet_ntp_dynamic/UFPR05/checkpoint',
               '/mnt/images/vitorhugo/test_results/mdl_rsnt50_ntp_static/UFPR05/checkpoint',
               '/mnt/images/vitorhugo/test_results/mdl_rsnt50_ntp_dynamic/UFPR05/checkpoint',
               '/mnt/images/vitorhugo/test_results/model_3cvnn/PUC/checkpoint',
               '/mnt/images/vitorhugo/test_results/mdl_mbnetv3lg_ntp_static/PUC/checkpoint',
               '/mnt/images/vitorhugo/test_results/mdl_mbnetv3lg_ntp_dynamic/PUC/checkpoint',
               '/mnt/images/vitorhugo/test_results/mdl_efficientnet_ntp_static/PUC/checkpoint',
               '/mnt/images/vitorhugo/test_results/mdl_efficientnet_ntp_dynamic/PUC/checkpoint',
               '/mnt/images/vitorhugo/test_results/mdl_rsnt50_ntp_static/PUC/checkpoint',
               '/mnt/images/vitorhugo/test_results/mdl_rsnt50_ntp_dynamic/PUC/checkpoint']


for model_path in model_paths:

    if 'model_3cvnn' in PurePath(model_path).parts:
        img_size = 32
    else:
        img_size = 128

    for university in universities:

        dataset = None
        train_dataset = None
        test_dataset = None
        validation_dataset = None
        test_ds = None
        model = None
        results_file = None

        dataset = base.dataset_gen(path,train,test,validation,university)

        train_dataset, test_dataset, validation_dataset = dataset

        test_ds = tf.data.Dataset.from_generator(
                        base.dataset_generator,
                        args=[test_dataset, img_size, True],
                        output_signature=(
                            tf.TensorSpec(shape=(img_size,img_size,3), dtype=tf.uint8),
                            tf.TensorSpec(shape=(), dtype=tf.uint8)))

        model = tf.keras.models.load_model(model_path)

        loss, acc = model.evaluate(test_ds.batch(32).prefetch(4), verbose=2)

        print(f'loss: {loss},\n accuracy: {acc}')

        print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))

        restored_acc = ('Restored model, accuracy: {:5.2f}%'.format(100 * acc))


        results_file = open(f'{model_path}/results_valuation_{university}.txt', 'w')
        results_file.write(f'loss: {loss},\n accuracy: {acc} \n Restored model, accuracy: {restored_acc}')
        results_file.close()
