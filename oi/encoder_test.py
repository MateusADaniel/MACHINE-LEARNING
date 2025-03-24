import encoder
import base
import threecvnn
import logging
import tensorflow as tf

logger = logging.getLogger(__name__)

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p',
                       filename='log.txt', encoding='utf-8', level=logging.INFO)

train_dataset, test_dataset, validation_dataset = base.dataset_gen('./', 0.01, 0.001, 0.001, 'PUC', True)

shape = 64
#latent_dim = 64
#autoencoder = encoder.Autoencoder(latent_dim, shape)
autoencoder = threecvnn.encoder()
autoencoder.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError(),metrics=['accuracy'])

base.train_model(autoencoder, train_dataset, test_dataset, validation_dataset, './tests/decoder/checkpoint.keras', 64, './tests/decoder/')
