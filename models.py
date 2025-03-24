from tensorflow import keras
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.models import Model
from contextlib import redirect_stdout
from tensorflow.keras.utils import plot_model
import tensorflow as tf
import numpy as np
from PIL import Image
import base
import matplotlib.pyplot as plt
import io
import json
import pdb
import logging
import contextlib

# Olhar estes imports depois
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Sequential, Model, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Lambda
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.regularizers import l2
import tensorflow.keras.backend as K
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, brier_score_loss

class OutputLogger:
    def __init__(self, name="root", level="INFO"):
        self.logger = logging.getLogger(name)
        self.name = self.logger.name
        self.level = getattr(logging, level)
        self._redirector = contextlib.redirect_stdout(self)

    def write(self, msg):
        if msg and not msg.isspace():
            self.logger.log(self.level, msg)

    def flush(self): pass

    def __enter__(self):
        self._redirector.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # let contextlib do any exception handling here
        self._redirector.__exit__(exc_type, exc_value, traceback)

class ThreeCvnn(keras.Model):
    def __init__(self, num_classes=2, random=True, dataset=None, model_path=None, epochs=None, logger=None):
        super().__init__()

        self.logger = logger

        self.logger.info("Instanciando modelo ThreeCvnn com os parametros:")
        self.logger.info(f'num_classes: {num_classes}')
        self.logger.info(f'random: {random}')
        json.dump(dataset, open(f'{model_path}/dataset.json', 'w'), indent=4)
        self.logger.info(f'dataset: salvo em {model_path}/dataset.json')
        self.logger.info(f'model_path: {model_path}')
        self.logger.info(f'epochs: {epochs}')

        self.random = random
        self.dataset = dataset
        self.model_path = model_path
        self.epochs = epochs

        logger.info(
            f'Quantidade de dias de teste: {len(self.dataset["test"])}')
        logger.info(
            f'Quantidade de dias de treino: {len(self.dataset["train"])}')
        logger.info(
            f'Quantidade de dias de validation: {len(self.dataset["validation"])}')

        self.callback = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=f'{self.model_path}/best.weights.h5',
                monitor='val_accuracy',
                mode='max',
                save_best_only=True,
                save_weights_only=True),
            tf.keras.callbacks.CSVLogger(
                filename=f'{self.model_path}/training.log',
                separator=',',
                append=False)
        ]

        self.output_signature = (tf.TensorSpec(shape=(32, 32, 3), dtype=tf.float32),
                                 tf.TensorSpec(shape=(), dtype=tf.uint8))
        self.train_dataset = tf.data.Dataset.from_generator(
            self.generator,
            args=[list(self.dataset['train']), 32, True],
            output_signature=self.output_signature)
        self.test_dataset = tf.data.Dataset.from_generator(
            self.generator,
            args=[list(self.dataset['test']), 32, False],
            output_signature=self.output_signature)
        self.validation_dataset = tf.data.Dataset.from_generator(
            self.generator,
            args=[list(self.dataset['validation']), 32, False],
            output_signature=self.output_signature)

        self.block_0 = keras.layers.Conv2D(32, kernel_size=(
            3, 3), strides=(1, 1), activation='relu', padding='same')
        self.block_1 = keras.layers.MaxPool2D(
            pool_size=(2, 2), strides=(2, 2), padding='valid')
        self.block_2 = keras.layers.Conv2D(
            64, kernel_size=(3, 3), strides=(1, 1), padding='valid')
        self.block_3 = keras.layers.MaxPool2D(
            pool_size=(2, 2), strides=(2, 2), padding='valid')
        self.block_4 = keras.layers.Conv2D(
            64, kernel_size=(3, 3), strides=(1, 1), padding='valid')
        self.block_5 = keras.layers.MaxPool2D(
            pool_size=(2, 2), strides=(2, 2), padding='valid')
        self.block_6 = keras.layers.Flatten()
        self.block_7 = keras.layers.Dense(64, activation='relu')
        self.block_8 = keras.layers.Dense(2, activation='softmax')

        self.logger.info("Concluida instanciação do modelo ThreeCvnn")

    def call(self, inputs, training=False):
        x = self.block_0(inputs, training=training)
        x = self.block_1(x, training=training)
        x = self.block_2(x, training=training)
        x = self.block_3(x, training=training)
        x = self.block_4(x, training=training)
        x = self.block_5(x, training=training)
        x = self.block_6(x, training=training)
        x = self.block_7(x, training=training)
        return self.block_8(x, training=training)


    def train(self):

        self.logger.info("Iniciando treino do modelo ThreeCvnn")

        self.logger.info("Compilando o modelo")
        self.compile(optimizer='Adam',
                     loss=keras.losses.SparseCategoricalCrossentropy(
                         from_logits=True),
                     metrics=['accuracy'])
        self.logger.info("Modelo compilado")

        self.logger.info("Iniciando model.fit do ThreeCvnn")
        history = self.fit(x=self.train_dataset.batch(32).prefetch(4), epochs=self.epochs,
                           validation_data=self.validation_dataset.batch(
                               32).prefetch(4),
                           callbacks=self.callback)
        
        # summarize history for accuracy
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(f'{self.model_path}/accuracy.png')
        plt.close()
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(f'{self.model_path}/loss.png')
        plt.close()

        self.logger.info("Concluido model.fit")

        self.logger.info("Salvando o modelo")
        self.save(f'{self.model_path}/final_model.keras')
        self.logger.info(
            f'Modelo salvo em {self.model_path}/final_model.keras')
        with OutputLogger("my_logger", "WARN") as redirector:
            self.summary()

    def generator(self, data, img_size, random):

        idx = np.arange(len(data))

        if random:
            np.random.shuffle(idx)

        for i in idx:
            label = base.convert_label(data[i][1].decode("utf-8"))
            img = tf.keras.utils.load_img(data[i][0])

            img = img.resize((img_size, img_size))
            #img = tf.keras.utils.img_to_array(img)/255.0
            img = tf.keras.utils.img_to_array(img)

            yield img, label

    def visualization_best(self):
        self.load_weights(f'{self.model_path}/final_model.keras')
        data = iter(self.test_dataset.take(36))
        fig = plt.figure(figsize=(32, 32))
        rows = 6
        columns = 6
        for i in range(36):
            image, label = next(data)
            image_with_batch = np.expand_dims(image, axis=0)
            predicted_label = self.predict(image_with_batch)[0].tolist()

            prediction = None
            for j in predicted_label:
                if prediction == None or j > prediction:
                    prediction = j
                    prediction_pos = predicted_label.index(j)
                    prediction_pos = base.convert_label(prediction_pos)

            fig.add_subplot(rows, columns, i+1)
            plt.imshow(image)
            plt.axis("off")
            plt.title(f"Predicted: {prediction_pos} Truth: {label}")
        plt.savefig(f'{self.model_path}/best_prediction.png')
        plt.close()
    def visualization(self):
        data = iter(self.test_dataset.take(36))
        fig = plt.figure(figsize=(32, 32))
        rows = 6
        columns = 6
        for i in range(36):
            image, label = next(data)
            image_with_batch = np.expand_dims(image, axis=0)
            predicted_label = self.predict(image_with_batch)[0].tolist()

            prediction = None
            for j in predicted_label:
                if prediction == None or j > prediction:
                    prediction = j
                    prediction_pos = predicted_label.index(j)
                    prediction_pos = base.convert_label(prediction_pos)

            fig.add_subplot(rows, columns, i+1)
            plt.imshow(image)
            plt.axis("off")
            plt.title(f"Predicted: {prediction_pos} Truth: {label}")
        plt.savefig(f'{self.model_path}/final_prediction.png')
        plt.close()

class Whale(keras.Model):
    def __init__(self, num_classes=2, random=True, dataset=None, model_path=None, epochs=None, logger=None):
        super().__init__()

        self.logger = logger

        self.logger.info(
            "Instanciando modelo Whale com os parametros:")
        self.logger.info(f'num_classes: {num_classes}')
        self.logger.info(f'random: {random}')
        json.dump(dataset, open(f'{model_path}/dataset.json', 'w'), indent=4)
        self.logger.info(f'dataset: salvo em {model_path}/dataset.txt')
        self.logger.info(f'model_path: {model_path}')
        self.logger.info(f'epochs: {epochs}')

        self.random = random
        self.dataset = dataset
        self.model_path = model_path
        self.epochs = epochs

        logger.info(
            f'Quantidade de dias de teste: {len(self.dataset["test"])}')
        logger.info(
            f'Quantidade de dias de treino: {len(self.dataset["train"])}')
        logger.info(
            f'Quantidade de dias de validation: {len(self.dataset["validation"])}')

        self.callback = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=f'{self.model_path}/best.weights.h5',
                monitor='val_accuracy',
                mode='max',
                save_best_only=True,
                save_weights_only=True),
            tf.keras.callbacks.CSVLogger(
                filename=f'{self.model_path}/training.log',
                separator=',',
                append=False)
        ]

        self.output_signature = (tf.TensorSpec(shape=(32, 32, 3), dtype=tf.float32),
                                 tf.TensorSpec(shape=(32, 32, 3), dtype=tf.float32))

        self.train_dataset = tf.data.Dataset.from_generator(
            self.generator,
            args=[list(self.dataset['train']), 32, True],
            output_signature=self.output_signature)
        self.test_dataset = tf.data.Dataset.from_generator(
            self.generator,
            args=[list(self.dataset['test']), 32, False],
            output_signature=self.output_signature)
        self.validation_dataset = tf.data.Dataset.from_generator(
            self.generator,
            args=[list(self.dataset['validation']), 32, False],
            output_signature=self.output_signature)

        self.encoder = tf.keras.Sequential([
            keras.layers.Conv2D(192, kernel_size=(3, 3), strides=(
                1, 1), activation='relu', padding='same'),
            keras.layers.MaxPool2D(pool_size=(
                2, 2), strides=(2, 2), padding='valid'),
            keras.layers.Conv2D(64, kernel_size=(
                3, 3), strides=(1, 1), padding='valid', activation='relu'),
            keras.layers.MaxPool2D(pool_size=(
                2, 2), strides=(2, 2), padding='valid'),
            keras.layers.Conv2D(64, kernel_size=(
                3, 3), strides=(1, 1), padding='valid', activation='relu'),
            keras.layers.MaxPool2D(pool_size=(
                2, 2), strides=(2, 2), padding='valid'),
            keras.layers.Flatten(),
            keras.layers.Dense(64, activation='relu')
        ],
            name='encoder'
        )
        self.decoder = tf.keras.Sequential([
            keras.layers.Dense(64, activation='relu'),
            # Shape to match the output of the last MaxPool2D layer in encoder
            keras.layers.Reshape((4, 4, 4)),
            keras.layers.Conv2DTranspose(64, kernel_size=(
                3, 3), strides=(2, 2), padding='same', activation='relu'),
            keras.layers.Conv2DTranspose(64, kernel_size=(
                3, 3), strides=(2, 2), padding='same', activation='relu'),
            keras.layers.Conv2DTranspose(32, kernel_size=(
                3, 3), strides=(2, 2), padding='same', activation='relu'),
            keras.layers.Conv2DTranspose(3, kernel_size=(3, 3), strides=(
                1, 1), padding='same', activation='sigmoid')
        ],
            name='decoder'
        )

        self.logger.info("Concluida instanciação do modelo ThreeCvnn_encoder")

    def call(self, inputs, training=False):
        encoded = self.encoder(inputs, training=training)
        decoded = self.decoder(encoded, training=training)
        return decoded

    def generator(self, data, img_size, random):

        idx = np.arange(len(data))

        if random:
            np.random.shuffle(idx)

        for i in idx:
            img = tf.keras.utils.load_img(data[i][0])

            #img = img.convert('L')
            img = img.resize((img_size, img_size))
            #img = tf.keras.utils.img_to_array(img)/255.0
            img = tf.keras.utils.img_to_array(img)

            yield img, img

    def train(self):
        self.logger.info("Iniciando treino do modelo ThreeCvnn_encoder")

        self.logger.info("Compilando o modelo")
        self.compile(optimizer='Adam',
                     loss=keras.losses.MeanAbsoluteError(),
                     metrics=['accuracy'])
        self.logger.info("Modelo compilado")

        self.logger.info("Iniciando model.fit do ThreeCvnn_encoder")
        history = self.fit(x=self.train_dataset.batch(32).prefetch(4), epochs=self.epochs,
                           validation_data=self.validation_dataset.batch(
                               32).prefetch(4),
                           callbacks=self.callback)
        # summarize history for accuracy
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(f'{self.model_path}/accuracy.png')
        plt.close()
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(f'{self.model_path}/loss.png')
        plt.close()
        self.logger.info("Concluido model.fit")

        self.logger.info("Salvando o modelo")
        self.save_weights(f'{self.model_path}/final_model.weights.h5')
        self.logger.info(
            f'Modelo salvo em {self.model_path}/final_model.weights.h5')
        with OutputLogger("my_logger", "WARN") as redirector:
            self.summary()

    def visualization_best(self):
        self.load_weights(f'{self.model_path}/final_model.weights.h5')
        images = iter(self.test_dataset.take(9))
        fig = plt.figure(figsize=(32, 32))
        rows = 2
        columns = 9
        for i in range(9):
            image, _ = next(images)
            image_with_batch = np.expand_dims(image, axis=0)
            image_generated = tf.keras.utils.array_to_img(
                np.squeeze(self(image_with_batch), 0))
            fig.add_subplot(rows, columns, i+1)
            plt.imshow(image_generated)
            plt.axis("off")
            plt.title("Generated")
            fig.add_subplot(rows, columns, i+10)
            plt.imshow(tf.keras.utils.array_to_img(image))
            plt.axis("off")
            plt.title("Original")
        plt.savefig(f'{self.model_path}/best_prediction.png')
        plt.close()
    def visualization(self):
        images = iter(self.test_dataset.take(9))
        fig = plt.figure(figsize=(32, 32))
        rows = 2
        columns = 9
        for i in range(9):
            image, _ = next(images)
            image_with_batch = np.expand_dims(image, axis=0)
            image_generated = tf.keras.utils.array_to_img(
                np.squeeze(self(image_with_batch), 0))
            fig.add_subplot(rows, columns, i+1)
            plt.imshow(image_generated)
            plt.axis("off")
            plt.title("Generated")
            fig.add_subplot(rows, columns, i+10)
            plt.imshow(tf.keras.utils.array_to_img(image))
            plt.axis("off")
            plt.title("Original")
        plt.savefig(f'{self.model_path}/final_prediction.png')
        plt.close()

class ThreeCvnn_Encoder(keras.Model):
    def __init__(self, num_classes=2, random=True, dataset=None, model_path=None, epochs=None, dropout=False, logger=None):
        super().__init__()

        if logger != None:
            self.logger = logger

            self.logger.info(
                "Instanciando modelo ThreeCvnn_Encoder com os parametros:")
            self.logger.info(f'num_classes: {num_classes}')
            self.logger.info(f'random: {random}')
            json.dump(dataset, open(f'{model_path}/dataset.json', 'w'), indent=4)
            self.logger.info(f'dataset: salvo em {model_path}/dataset.txt')
            self.logger.info(f'model_path: {model_path}')
            self.logger.info(f'epochs: {epochs}')
            self.logger.info(f'dropout: {dropout}')

        self.random = random
        self.dataset = dataset
        self.model_path = model_path
        self.epochs = epochs
        self.dropout = bool(dropout)

        if logger != None:
            logger.info(
                f'Quantidade de dias de teste: {len(self.dataset["test"])}')
            logger.info(
                f'Quantidade de dias de treino: {len(self.dataset["train"])}')
            logger.info(
                f'Quantidade de dias de validation: {len(self.dataset["validation"])}')

        self.callback = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=f'{self.model_path}/best.weights.h5',
                monitor='val_accuracy',
                mode='max',
                save_best_only=True,
                save_weights_only=True),
            tf.keras.callbacks.CSVLogger(
                filename=f'{self.model_path}/training.log',
                separator=',',
                append=False)
        ]

        self.output_signature = (tf.TensorSpec(shape=(32, 32, 3), dtype=tf.float32),
                                 tf.TensorSpec(shape=(32, 32, 3), dtype=tf.float32))

        self.train_dataset = tf.data.Dataset.from_generator(
            self.generator,
            args=[list(self.dataset['train']), 32, True],
            output_signature=self.output_signature)
        self.test_dataset = tf.data.Dataset.from_generator(
            self.generator,
            args=[list(self.dataset['test']), 32, False],
            output_signature=self.output_signature)
        self.validation_dataset = tf.data.Dataset.from_generator(
            self.generator,
            args=[list(self.dataset['validation']), 32, False],
            output_signature=self.output_signature)

        if self.dropout:
            self.encoder = tf.keras.Sequential([
                keras.layers.Conv2D(32, kernel_size=(3, 3), strides=(
                    1, 1), activation='relu', padding='same'),
                keras.layers.MaxPool2D(pool_size=(
                    2, 2), strides=(2, 2), padding='valid'),
                keras.layers.Conv2D(64, kernel_size=(
                    3, 3), strides=(1, 1), padding='valid', activation='relu'),
                keras.layers.MaxPool2D(pool_size=(
                    2, 2), strides=(2, 2), padding='valid'),
                keras.layers.Conv2D(64, kernel_size=(
                    3, 3), strides=(1, 1), padding='valid', activation='relu'),
                keras.layers.MaxPool2D(pool_size=(
                    2, 2), strides=(2, 2), padding='valid'),
                keras.layers.Flatten(),
                keras.layers.Dropout(0.3),
                keras.layers.Dense(64, activation='relu')
            ],
                name='encoder'
            )
        else:
            self.encoder = tf.keras.Sequential([
                keras.layers.Conv2D(32, kernel_size=(3, 3), strides=(
                    1, 1), activation='relu', padding='same'),
                keras.layers.MaxPool2D(pool_size=(
                    2, 2), strides=(2, 2), padding='valid'),
                keras.layers.Conv2D(64, kernel_size=(
                    3, 3), strides=(1, 1), padding='valid', activation='relu'),
                keras.layers.MaxPool2D(pool_size=(
                    2, 2), strides=(2, 2), padding='valid'),
                keras.layers.Conv2D(64, kernel_size=(
                    3, 3), strides=(1, 1), padding='valid', activation='relu'),
                keras.layers.MaxPool2D(pool_size=(
                    2, 2), strides=(2, 2), padding='valid'),
                keras.layers.Flatten(),
                keras.layers.Dense(64, activation='relu')
            ],
                name='encoder'
            )
        self.decoder = tf.keras.Sequential([
            keras.layers.Dense(64, activation='relu'),
            # Shape to match the output of the last MaxPool2D layer in encoder
            keras.layers.Reshape((4, 4, 4)),
            keras.layers.Conv2DTranspose(64, kernel_size=(
                3, 3), strides=(2, 2), padding='same', activation='relu'),
            keras.layers.Conv2DTranspose(64, kernel_size=(
                3, 3), strides=(2, 2), padding='same', activation='relu'),
            keras.layers.Conv2DTranspose(32, kernel_size=(
                3, 3), strides=(2, 2), padding='same', activation='relu'),
            keras.layers.Conv2DTranspose(3, kernel_size=(3, 3), strides=(
                1, 1), padding='same', activation='sigmoid')
        ],
            name='decoder'
        )

        if logger != None:
            self.logger.info("Concluida instanciação do modelo ThreeCvnn_encoder")

    def call(self, inputs, training=False):
        encoded = self.encoder(inputs, training=training)
        decoded = self.decoder(encoded, training=training)
        return decoded

    def generator(self, data, img_size, random):

        idx = np.arange(len(data))

        if random:
            np.random.shuffle(idx)

        for i in idx:
            img = tf.keras.utils.load_img(data[i][0])

            #img = img.convert('L')
            img = img.resize((img_size, img_size))
            img = tf.keras.utils.img_to_array(img)/255.0
            #img = tf.keras.utils.img_to_array(img)

            yield img, img

    def train(self):
        if logger != None:
            self.logger.info("Iniciando treino do modelo ThreeCvnn_encoder")

            self.logger.info("Compilando o modelo")
        self.compile(optimizer='Adam',
                     loss=keras.losses.MeanAbsoluteError(),
                     metrics=['accuracy'])
        if logger != None:
            self.logger.info("Modelo compilado")

            self.logger.info("Iniciando model.fit do ThreeCvnn_encoder")
        history = self.fit(x=self.train_dataset.batch(32).prefetch(4), epochs=self.epochs,
                           validation_data=self.validation_dataset.batch(
                               32).prefetch(4),
                           callbacks=self.callback)
        # summarize history for accuracy
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(f'{self.model_path}/accuracy.png')
        plt.close()
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(f'{self.model_path}/loss.png')
        plt.close()
        if logger != None:
            self.logger.info("Concluido model.fit")

            self.logger.info("Salvando o modelo")
            self.save_weights(f'{self.model_path}/final_model.weights.h5')
            self.logger.info(
                f'Modelo salvo em {self.model_path}/final_model.weights.h5')
        with OutputLogger("my_logger", "WARN") as redirector:
            self.summary()

    def visualization_best(self):
        images = self.test_dataset.take(9)
        images = images.repeat(2)
        images = iter(images)
        self.load_weights(f'{self.model_path}/final_model.weights.h5')
        fig = plt.figure(figsize=(32, 32))
        rows = 2
        columns = 9
        for i in range(9):
            image, _ = next(images)
            image_with_batch = np.expand_dims(image, axis=0)
            image_generated = np.squeeze(self(image_with_batch), 0)
            image_generated = tf.keras.utils.array_to_img(image_generated)
            fig.add_subplot(rows, columns, i+1)
            plt.imshow(image_generated)
            plt.axis("off")
            plt.title("Generated")
            fig.add_subplot(rows, columns, i+10)
            plt.imshow(tf.keras.utils.array_to_img(image))
            plt.axis("off")
            plt.title("Original")
        plt.savefig(f'{self.model_path}/best_prediction.png')
        plt.close()
    def visualization(self):
        images = iter(self.test_dataset.take(9))
        fig = plt.figure(figsize=(32, 32))
        rows = 2
        columns = 9
        for i in range(9):
            image, _ = next(images)
            image_with_batch = np.expand_dims(image, axis=0)
            image_generated = np.squeeze(self(image_with_batch), 0)
            image_generated = tf.keras.utils.array_to_img(image_generated)
            fig.add_subplot(rows, columns, i+1)
            plt.imshow(image_generated)
            plt.axis("off")
            plt.title("Generated")
            fig.add_subplot(rows, columns, i+10)
            plt.imshow(tf.keras.utils.array_to_img(image))
            plt.axis("off")
            plt.title("Original")
        plt.savefig(f'{self.model_path}/final_prediction.png')
        plt.close()

class ThreeCvnnClassifier(keras.Model):
    def __init__(self, num_classes=2, random=True, dataset=None, model_path=None, epochs=None, pretrained_weights=None, logger=None):
        super().__init__()

        if logger != None:
            self.logger = logger

            self.logger.info(
                "Instanciando modelo ThreeCvnnClassifier com os parametros:")
            self.logger.info(f'num_classes: {num_classes}')
            self.logger.info(f'random: {random}')
            json.dump(dataset, open(f'{model_path}/dataset.json', 'w'), indent=4)
            self.logger.info(f'dataset: salvo em {model_path}/dataset.txt')
            self.logger.info(f'model_path: {model_path}')
            self.logger.info(f'epochs: {epochs}')

        self.random = random
        self.dataset = dataset
        self.model_path = model_path
        self.epochs = epochs
        self.pretrained_weights = pretrained_weights

        if logger != None:
            logger.info(
                f'Quantidade de dias de teste: {len(self.dataset["test"])}')
            logger.info(
                f'Quantidade de dias de treino: {len(self.dataset["train"])}')
            logger.info(
                f'Quantidade de dias de validation: {len(self.dataset["validation"])}')

        self.callback = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=f'{self.model_path}/best.weights.h5',
                monitor='val_accuracy',
                mode='max',
                save_best_only=True,
                save_weights_only=True),
            tf.keras.callbacks.CSVLogger(
                filename=f'{self.model_path}/training.log',
                separator=',',
                append=False)
        ]

        self.output_signature = (tf.TensorSpec(shape=(32, 32, 3), dtype=tf.float32),
                                 tf.TensorSpec(shape=(), dtype=tf.uint8))

        self.train_dataset = tf.data.Dataset.from_generator(
            self.generator,
            args=[list(self.dataset['train']), 32, True],
            output_signature=self.output_signature)
        self.test_dataset = tf.data.Dataset.from_generator(
            self.generator,
            args=[list(self.dataset['test']), 32, False],
            output_signature=self.output_signature)
        self.validation_dataset = tf.data.Dataset.from_generator(
            self.generator,
            args=[list(self.dataset['validation']), 32, False],
            output_signature=self.output_signature)

        self.base_model = ThreeCvnn_Encoder(
            num_classes=num_classes, random=random, dataset=dataset, model_path=model_path, epochs=epochs, logger=logger)
        if pretrained_weights != None:
            self.base_model.load_weights(pretrained_weights)
        self.base_model.trainable = True
        self.base_model.layers.pop()

        self.dense = tf.keras.Sequential(
            layers=[
                keras.layers.Flatten(),
                keras.layers.Dropout(0.2),
                keras.layers.Dense(64, activation='relu'),
                keras.layers.Dense(2, activation='softmax')
            ],
            name='dense'
        )

        with OutputLogger("my_logger", "WARN") as redirector:
            self.summary()

        if logger != None:
            self.logger.info(
                "Concluida instanciação do modelo ThreeCvnnClassifier")

    def call(self, x, training=False):
        x = self.base_model(x, training=training)
        x = self.dense(x, training=training)
        return x

    def generator(self, data, img_size, random):

        idx = np.arange(len(data))

        if random:
            np.random.shuffle(idx)

        for i in idx:
            label = base.convert_label(data[i][1].decode("utf-8"))
            img = tf.keras.utils.load_img(data[i][0])

            #img = img.convert('L')
            img = img.resize((img_size, img_size))
            # img = tf.keras.utils.img_to_array(img)/255.0
            img = tf.keras.utils.img_to_array(img)

            yield img, label

    def train(self):
        if logger != None:
            self.logger.info("Iniciando treino do modelo ThreeCvnnClassifier")

            self.logger.info("Compilando o modelo")
            self.compile(optimizer='Adam',
                         loss=keras.losses.SparseCategoricalCrossentropy(
                             from_logits=True),
                         metrics=['accuracy'])
            self.logger.info("Modelo compilado")

            self.logger.info("Iniciando model.fit do ThreeCvnnClassifier")
        history = self.fit(x=self.train_dataset.batch(32).prefetch(4), epochs=self.epochs,
                           validation_data=self.validation_dataset.batch(
                               32).prefetch(4),
                           callbacks=self.callback)
        # summarize history for accuracy
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.savefig(f'{self.model_path}/accuracy.png')
        plt.close()
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.savefig(f'{self.model_path}/loss.png')
        plt.close()
        if logger != None:
            self.logger.info("Concluido model.fit")

            self.logger.info("Salvando o modelo")
        self.save_weights(f'{self.model_path}/final_model.weights.h5')
        if logger != None:
            self.logger.info(
                f'Modelo salvo em {self.model_path}/final_model.weights.h5')
        with OutputLogger("my_logger", "WARN") as redirector:
            self.summary()

    def visualization_best(self):
        data = self.test_dataset.take(36)
        data = data.repeat(2)
        data = iter(data)
        self.load_weights(f'{self.model_path}/final_model.weights.h5')
        fig = plt.figure(figsize=(32, 32))
        rows = 6
        columns = 6
        for i in range(36):
            image, label = next(data)
            image_with_batch = np.expand_dims(image, axis=0)
            predicted_label = self.predict(image_with_batch)[0].tolist()
            #predicted_label = self.predict(image_with_batch)[0]

            prediction = None
            for j in predicted_label:
                if prediction == None or j > prediction:
                    prediction = j
                    prediction_pos = predicted_label.index(j)
                    prediction_pos = base.convert_label(prediction_pos)

            fig.add_subplot(rows, columns, i+1)
            #plt.imshow(image)
            plt.imshow(tf.keras.utils.array_to_img(image))
            plt.axis("off")
            plt.title(f"Predicted: {prediction_pos} Truth: {label}")
        plt.savefig(f'{self.model_path}/best_prediction.png')
        plt.close()
    def visualization(self):
        data = iter(self.test_dataset.take(36))
        fig = plt.figure(figsize=(32, 32))
        rows = 6
        columns = 6
        for i in range(36):
            image, label = next(data)
            image_with_batch = np.expand_dims(image, axis=0)
            predicted_label = self.predict(image_with_batch)[0].tolist()
            #predicted_label = self.predict(image_with_batch)[0]

            prediction = None
            for j in predicted_label:
                if prediction == None or j > prediction:
                    prediction = j
                    prediction_pos = predicted_label.index(j)
                    prediction_pos = base.convert_label(prediction_pos)

            fig.add_subplot(rows, columns, i+1)
            #plt.imshow(image)
            plt.imshow(tf.keras.utils.array_to_img(image))
            plt.axis("off")
            plt.title(f"Predicted: {prediction_pos} Truth: {label}")
        plt.savefig(f'{self.model_path}/final_prediction.png')
        plt.close()

class Simple(keras.Model):
    def __init__(self, num_classes=2, random=True, dataset=None, model_path=None, epochs=None, dropout=False, logger=None):
        super().__init__()

        self.logger = logger

        self.logger.info("Instanciando modelo Simple com os parametros:")
        self.logger.info(f'num_classes: {num_classes}')
        self.logger.info(f'random: {random}')
        json.dump(dataset, open(f'{model_path}/dataset.json', 'w'), indent=4)
        self.logger.info(f'dataset: salvo em {model_path}/dataset.txt')
        self.logger.info(f'model_path: {model_path}')
        self.logger.info(f'epochs: {epochs}')
        self.logger.info(f'dropout: {dropout}')

        self.random = random
        self.dataset = dataset
        self.model_path = model_path
        self.epochs = epochs
        self.dropout = bool(dropout)

        logger.info(
            f'Quantidade de dias de teste: {len(self.dataset["test"])}')
        logger.info(
            f'Quantidade de dias de treino: {len(self.dataset["train"])}')
        logger.info(
            f'Quantidade de dias de validation: {len(self.dataset["validation"])}')

        self.callback = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=f'{self.model_path}/best.weights.h5',
                monitor='val_accuracy',
                mode='max',
                save_best_only=True,
                save_weights_only=True),
            tf.keras.callbacks.CSVLogger(
                filename=str(f'{self.model_path}/training.log'),
                separator=',',
                append=False)
        ]

        self.output_signature = (tf.TensorSpec(shape=(32, 32, 3), dtype=tf.float32),
                                 tf.TensorSpec(shape=(32, 32, 3), dtype=tf.float32))
        self.train_dataset = tf.data.Dataset.from_generator(
            self.generator,
            args=[list(self.dataset['train']), 32, True],
            output_signature=self.output_signature)
        self.test_dataset = tf.data.Dataset.from_generator(
            self.generator,
            args=[list(self.dataset['test']), 32, False],
            output_signature=self.output_signature)
        self.validation_dataset = tf.data.Dataset.from_generator(
            self.generator,
            args=[list(self.dataset['validation']), 32, False],
            output_signature=self.output_signature)

        self.latent_dim = 8
        self.shape = (32, 32, 3)

        if self.dropout:
            self.encoder = tf.keras.Sequential([
                keras.layers.Flatten(),
                keras.layers.Dense(self.latent_dim, activation='relu'),
                keras.layers.Dropout(0.3)
            ],
                name='encoder')
        else:
            self.encoder = tf.keras.Sequential([
                keras.layers.Flatten(),
                keras.layers.Dense(self.latent_dim, activation='relu'),
            ],
                name='encoder')

        self.decoder = tf.keras.Sequential([
            keras.layers.Dense(tf.math.reduce_prod(
                self.shape).numpy(), activation='sigmoid'),
            keras.layers.Reshape(self.shape)
        ],
            name='decoder')

        self.logger.info("Concluida instanciação do modelo Simple")

    def call(self, inputs, training=False):
        encoded = self.encoder(inputs, training=training)
        decoded = self.decoder(encoded, training=training)
        return decoded

    def generator(self, data, img_size, random):

        idx = np.arange(len(data))

        if random:
            np.random.shuffle(idx)

        for i in idx:
            img = tf.keras.utils.load_img(data[i][0])

            img = img.resize((img_size, img_size))
            # img = tf.keras.utils.img_to_array(img)/255.0
            img = tf.keras.utils.img_to_array(img)

            yield img, img

    def train(self):
        self.logger.info("Iniciando treino do modelo Simple")

        self.logger.info("Compilando o modelo")
        self.compile(optimizer='Adam',
                     loss=keras.losses.MeanAbsoluteError(),
                     metrics=['accuracy'])
        self.logger.info("Modelo compilado")

        self.logger.info("Iniciando model.fit do Simple")
        history = self.fit(x=self.train_dataset.batch(32).prefetch(4), epochs=self.epochs,
                           validation_data=self.validation_dataset.batch(
                               32).prefetch(4),
                           callbacks=self.callback)
        # summarize history for accuracy
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.savefig(f'{self.model_path}/accuracy.png')
        plt.close()
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.savefig(f'{self.model_path}/loss.png')
        plt.close()
        self.logger.info("Concluido model.fit")

        self.logger.info("Salvando o modelo")
        self.save(f'{self.model_path}/final_model.keras')
        self.logger.info(f'Modelo salvo em {self.model_path}/final_model.keras')
        with OutputLogger("my_logger", "WARN") as redirector:
            self.summary()

    def visualization_best(self):
        images = self.test_dataset.take(9)
        images = images.repeat(2)
        images = iter(images)
        fig = plt.figure(figsize=(32, 32))
        rows = 2
        columns = 9
        for i in range(9):
            image, _ = next(images)
            image_with_batch = np.expand_dims(image, axis=0)
            image_generated = np.squeeze(self(image_with_batch), 0)
            image_generated = tf.keras.utils.array_to_img(image_generated)
            fig.add_subplot(rows, columns, i+1)

            plt.imshow(image_generated)
            plt.axis("off")
            plt.title("Generated")
            fig.add_subplot(rows, columns, i+10)
            plt.imshow(tf.keras.utils.array_to_img(image))
            plt.axis("off")
            plt.title("Original")
        plt.savefig(f'{self.model_path}/final_prediction.png')
        plt.close()
        self.load_weights(f'{self.model_path}/final_model.keras')
        fig = plt.figure(figsize=(32, 32))
        rows = 2
        columns = 9
        for i in range(9):
            image, _ = next(images)
            image_with_batch = np.expand_dims(image, axis=0)
            image_generated = np.squeeze(self(image_with_batch), 0)
            image_generated = tf.keras.utils.array_to_img(image_generated)
            fig.add_subplot(rows, columns, i+1)

            plt.imshow(image_generated)
            plt.axis("off")
            plt.title("Generated")
            fig.add_subplot(rows, columns, i+10)
            plt.imshow(tf.keras.utils.array_to_img(image))
            plt.axis("off")
            plt.title("Original")
        plt.savefig(f'{self.model_path}/best_prediction.png')
        plt.close()
    def visualization(self):
        images = iter(self.test_dataset.take(9))
        fig = plt.figure(figsize=(32, 32))
        rows = 2
        columns = 9
        for i in range(9):
            image, _ = next(images)
            image_with_batch = np.expand_dims(image, axis=0)
            image_generated = np.squeeze(self(image_with_batch), 0)
            image_generated = tf.keras.utils.array_to_img(image_generated)
            fig.add_subplot(rows, columns, i+1)

            plt.imshow(image_generated)
            plt.axis("off")
            plt.title("Generated")
            fig.add_subplot(rows, columns, i+10)
            plt.imshow(tf.keras.utils.array_to_img(image))
            plt.axis("off")
            plt.title("Original")
        plt.savefig(f'{self.model_path}/final_prediction.png')
        plt.close()
class SiameseTraining(keras.Model):
    def __init__(self, num_classes=2, random=True, dataset=None, model_path=None, epochs=None, dropout=False, logger=None):
        super(SiameseTraining, self).__init__()
        self.input_shape = (105,105,1)

        self.logger = logger
        self.logger.info("Instanciando modelo SiameseTraining com os parametros:")
        self.logger.info(f'num_classes: {num_classes}')
        self.logger.info(f'random: {random}')
        json.dump(dataset, open(f'{model_path}/dataset.json', 'w'), indent=4)
        self.logger.info(f'dataset: salvo em {model_path}/dataset.json')
        self.logger.info(f'model_path: {model_path}')
        self.logger.info(f'epochs: {epochs}')

        self.random = random
        self.dataset = dataset
        self.model_path = model_path
        self.epochs = epochs
        self.dropout = bool(dropout)

        logger.info(f'Quantidade de dias de teste: {len(self.dataset["test"])}')
        logger.info(f'Quantidade de dias de treino: {len(self.dataset["train"])}')
        logger.info(f'Quantidade de dias de validation: {len(self.dataset["validation"])}')

        self.callback = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=f'{self.model_path}/best.weights.h5',
                monitor='val_accuracy',
                mode='max',
                save_best_only=True,
                save_weights_only=True),
            tf.keras.callbacks.CSVLogger(
                filename=f'{self.model_path}/training_inputs.log',
                separator=',',
                append=False)
        ]

        self.output_signature = ((tf.TensorSpec(shape=(105, 105, 1), dtype=tf.float32),
                                tf.TensorSpec(shape=(105, 105, 1), dtype=tf.float32)),
                                tf.TensorSpec(shape=(), dtype=tf.float32))

        self.filter_training_dataset('train', 51, True)
        self.filter_training_dataset('test', 67, True) # sempre colocar +3 no numero desejado em cada classe
        self.filter_training_dataset('validation', 20, True)

        logger.info(f'Quantidade de dias de teste depois do ajuste: {len(self.dataset["test"])}')
        logger.info(f'Quantidade de dias de treino depois do ajuste: {len(self.dataset["train"])}')
        logger.info(f'Quantidade de dias de validation depois do ajuste: {len(self.dataset["validation"])}')

        counts = self.count_images_per_class()
        self.logger.info(f"Contagem de imagens por classe: {counts}")

        self.reference_pairs_train = self.generate_pairs('train', 105, True)
        self.reference_pairs_validation = self.generate_pairs('validation', 105, True)
        self.reference_pairs_test = self.generate_pairs('test', 105, True)

        # Definicao dos datasets

        self.train_dataset = tf.data.Dataset.from_generator(
            self.generator,
            args=[self.dataset['train'], 105, True, self.reference_pairs_train],
            output_signature=self.output_signature)
            
        self.test_dataset = tf.data.Dataset.from_generator(
            self.generator_test,
            args=[self.dataset['test'], 105, True, self.reference_pairs_test],
            output_signature=self.output_signature)
    
        #pdb.set_trace()
        self.validation_dataset = tf.data.Dataset.from_generator(
            self.generator,
            args=[self.dataset['validation'], 105, True, self.reference_pairs_validation],
            output_signature=self.output_signature)
    


        # Defina a arquitetura CNN
        self.base_model = tf.keras.Sequential([ 
            keras.layers.Conv2D(64, kernel_size=(10, 10), activation='relu', input_shape=(105, 105, 1)),
            keras.layers.MaxPooling2D(),
            keras.layers.Conv2D(128, kernel_size=(7, 7), activation='relu'),
            keras.layers.MaxPooling2D(),
            keras.layers.Conv2D(128, kernel_size=(4, 4), activation='relu'),
            keras.layers.MaxPooling2D(),
            keras.layers.Conv2D(256, kernel_size=(4, 4), activation='relu'),
            keras.layers.Flatten(),
            keras.layers.Dense(4096, activation='sigmoid')
        ],
            name='base_model'
        )
        
        # Camada de previsão
        self.dense = keras.layers.Dense(1, activation='sigmoid')


    def call(self, inputs, training=False):
        input1,input2 = inputs
        #pdb.set_trace()
        right_encoded = self.base_model(input1, training=training)
        left_encoded = self.base_model(input2, training=training)
        
        distance = tf.abs(right_encoded - left_encoded)
        output = self.dense(distance, training=training)
        return output
        
    def generate_pairs(self, datatype, img_size, random):
        output = []
        data = self.dataset[datatype]
        idx = np.arange(len(data))
        if random:
            np.random.shuffle(idx)
        while len(output) != 3:
            idx = np.arange(len(data))
            np.random.shuffle(idx)
            index, idx = idx[-1], idx[:-1]
            label = data[index][1]
            if label == 'Empty':
                img = data[index][0]
                output.append([img, label])
                data.pop(index)

        idx = np.arange(len(data))
        np.random.shuffle(idx)
        while len(output) != 6:
            idx = np.arange(len(data))
            np.random.shuffle(idx)
            index, idx = idx[-1], idx[:-1]
            label = data[index][1]
            if label == 'Occupied':
                img = data[index][0]
                output.append([img, label])
                data.pop(index)
        self.dataset[datatype] = data
        return output

    def generator(self, data, img_size, random, pairs):
        #data_augmentation = tf.keras.Sequential([
        #    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        #    tf.keras.layers.RandomRotation(0.4),
        #])
        idx = np.arange(len(data))
        if random:
            np.random.shuffle(idx)
        for i in idx:
            n = 0
            while n != 6:
                pair_label = pairs[n][1]
                pair_image = pairs[n][0]
                pair_image = tf.keras.utils.load_img(pair_image, color_mode='grayscale')
                pair_image = pair_image.resize((img_size, img_size))
                pair_image = tf.keras.utils.img_to_array(pair_image) / 255.0
                #pair_image = data_augmentation(pair_image)
                
                image_label = data[i][1]
                image = tf.keras.utils.load_img(data[i][0], color_mode='grayscale')

                image = image.resize((img_size, img_size))
                image = tf.keras.utils.img_to_array(image) / 255.0
                #image = data_augmentation(image)

                if image_label == pair_label:
                    label = 1.0
                else:
                    label = 0.0
                n += 1

                image_pair_certa = np.array(pair_image)
                image_certa = np.array(image)
                label_certa = np.array(label, dtype=np.float32)
                #pdb.set_trace()

                yield (image_certa, image_pair_certa), label_certa

    def generator_test(self, data, img_size, random, pairs):
        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal_and_vertical"),
            tf.keras.layers.RandomRotation(0.4),
        ])
        idx = np.arange(len(data))
        if random:
            np.random.shuffle(idx)

        pair_image_path = data[idx[0]][0]
        pair_label = data[idx[0]][1]
        
        pair_image = tf.keras.utils.load_img(pair_image_path, color_mode='grayscale')
        pair_image = pair_image.resize((img_size, img_size))
        pair_image = tf.keras.utils.img_to_array(pair_image) / 255.0

        for i in idx:
            
            #pair_image = data_augmentation(pair_image)


            image_label = data[i][1]
            image = tf.keras.utils.load_img(data[i][0], color_mode='grayscale')
            
            image = image.resize((img_size, img_size))
            image = tf.keras.utils.img_to_array(image) / 255.0
            image = data_augmentation(image)

            if image_label == pair_label:
                label = 1.0
            else:
                label = 0.0
            
            image_pair_certa = np.array(pair_image)
            image_certa = np.array(image)
            label_certa = np.array(label, dtype=np.float32)
            
            yield (image_certa, image_pair_certa), label_certa

        # 1 para empty 0 para occupied


#verificar a rede


    def train(self):

        # Compile o modelo
        self.logger.info("Compilando o modelo")
        self.compile(optimizer='Adam',
                    loss=keras.losses.BinaryCrossentropy(
                        from_logits=False),
                    metrics=['accuracy'])
        self.logger.info("Modelo compilado")
        self.logger = tf.get_logger()
        self.logger.info("Concluída a instanciação do modelo SiameseNetwork")
        with OutputLogger("my_logger", "WARN") as redirector:
            self.summary()
        self.logger.info("Iniciando treino do modelo Siamese")

        self.logger.info("Iniciando model.fit do Siamese")
        history = self.fit(x=self.train_dataset.batch(32).prefetch(4), epochs=self.epochs,
                        validation_data=self.validation_dataset.batch(32).prefetch(4),
                        callbacks=self.callback
        )

        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.savefig(f'{self.model_path}/accuracy.png')
        plt.close()
#
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.savefig(f'{self.model_path}/loss.png')
        plt.close()
#
        self.logger.info("Concluido model.fit")
#
        with OutputLogger("my_logger", "WARN") as redirector:
            self.summary()
        #self.logger.info("Salvando o modelo")
        #self.save(f'{self.model_path}/final_model.keras')
        #self.logger.info(
        #    f'Modelo salvo em {self.model_path}/final_model.keras')

        #self.load_weights(f'{self.model_path}/final_model.keras')




    def visualization_best(self):
                # Load the model weights
        self.load_weights('./batch_results/ufpr052000/siamese/UFPR05_1/best.weights.h5')
        images = list(self.test_dataset.take(128))  # Pegue 18 pares para um total de 36 imagens
        true_labels = []
        predicted_labels = []
        self.image1 = images 

        fig = plt.figure(figsize=(32, 64))  # Ajuste o tamanho da figura conforme necessário
        rows = 50  # Quatro linhas
        columns = 10  # Nove colunas

        for i in range(128):
            img = images[i][0]  # Assumindo que os dados são uma tupla (imagem, rótulo)
            img = np.squeeze(img)  # Remove quaisquer dimensões extras, se necessário

            imagem1 = img[0]
            imagem2 = img[1]
            label = images[i][1]

            # Fazer previsões para o par de imagens
            imagem1_batch = np.expand_dims(imagem1, axis=0)
            imagem2_batch = np.expand_dims(imagem2, axis=0)

            predicted_label = self.predict([imagem1_batch, imagem2_batch])[0]
            
            true_labels.append(label)
            if predicted_label > 0.5:
                predicted_labels.append(1.0)
            else: 
                predicted_labels.append(0.0)


            
            # Plotar as imagens
    #        fig.add_subplot(rows, columns, 2*i+1)
    #        plt.imshow(imagem1, cmap='gray')
    #        plt.axis('off')
#
    #        fig.add_subplot(rows, columns, 2*i+2)
    #        plt.imshow(imagem2, cmap='gray')
    #        plt.axis('off')
#
    #        plt.title(f'P:{predicted_label.tolist()}, T:{label}, N: {predicted_labels[i]},')
#
    #    plt.tight_layout()
#
    #    # Salvar a figura
    #    plt.savefig(f'{self.model_path}/final_prediction_128_uPUC_2000.png')
    #    plt.close()
        
        # Calcular métricas de avaliação
        true_labels = np.array(true_labels)
        predicted_labels = np.array(predicted_labels)

        # Converter previsões para classes binárias com um limiar de 0.5
        binary_predictions = predicted_labels.astype(np.float32)

        accuracy = accuracy_score(true_labels, binary_predictions)
        precision = precision_score(true_labels, binary_predictions)
        recall = recall_score(true_labels, binary_predictions)
        f1 = f1_score(true_labels, binary_predictions)
        brier = brier_score_loss(true_labels, predicted_labels)

        print(f'Accuracy: {accuracy:.4f}')
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'F1 Score: {f1:.4f}')
        print(f'Brier Score: {brier:.4f}')


        self.logger.info(f"---------------------- REDE TREINADA COM PUC IMGENS---------------------------------------")

        self.logger.info(f"Accuracy: {accuracy:.4f}")
        self.logger.info(f'Precision: {precision:.4f}')
        self.logger.info(f'Recall: {recall:.4f}')
        self.logger.info(f'F1 Score: {f1:.4f}')
        self.logger.info(f'Brier Score: {brier:.4f}')

    def visualization_best2(self):
        # Load the model weights
        self.load_weights('./batch_results/UFPR045000/siamese/UFPR04_1/final_model.keras')
        images = self.image1  # Pegue 18 pares para um total de 36 imagens
        true_labels = []
        predicted_labels = []

        #fig = plt.figure(figsize=(32, 64))  # Ajuste o tamanho da figura conforme necessário
        #rows = 32  # Quatro linhas
        #columns = 8  # Nove colunas

        for i in range(128):
            img = images[i][0]  # Assumindo que os dados são uma tupla (imagem, rótulo)
            img = np.squeeze(img)  # Remove quaisquer dimensões extras, se necessário

            imagem1 = img[0]
            imagem2 = img[1]
            label = images[i][1]

            # Fazer previsões para o par de imagens
            imagem1_batch = np.expand_dims(imagem1, axis=0)
            imagem2_batch = np.expand_dims(imagem2, axis=0)

            predicted_label = self.predict([imagem1_batch, imagem2_batch])[0]
            
            true_labels.append(label)
            if predicted_label > 0.5:
                predicted_labels.append(1.0)
            else: 
                predicted_labels.append(0.0)


            
            # Plotar as imagens
            #fig.add_subplot(rows, columns, 2*i+1)
            #plt.imshow(imagem1, cmap='gray')
            #plt.axis('off')
#
            #fig.add_subplot(rows, columns, 2*i+2)
            #plt.imshow(imagem2, cmap='gray')
            #plt.axis('off')
#
            #plt.title(f'P:{predicted_label.tolist()}, T:{label}, N: {predicted_labels[i]},')

        #plt.tight_layout()
#
        ## Salvar a figura
        #plt.savefig(f'{self.model_path}/final_prediction_128_ufpr05_5000.png')
        #plt.close()
        
        # Calcular métricas de avaliação
        true_labels = np.array(true_labels)
        predicted_labels = np.array(predicted_labels)

        # Converter previsões para classes binárias com um limiar de 0.5
        binary_predictions = predicted_labels.astype(np.float32)

        accuracy = accuracy_score(true_labels, binary_predictions)
        precision = precision_score(true_labels, binary_predictions)
        recall = recall_score(true_labels, binary_predictions)
        f1 = f1_score(true_labels, binary_predictions)
        brier = brier_score_loss(true_labels, predicted_labels)

        print(f'Accuracy: {accuracy:.4f}')
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'F1 Score: {f1:.4f}')
        print(f'Brier Score: {brier:.4f}')
        self.logger.info(f"---------------------- REDE TREINADA COM UFPR04 IMGENS---------------------------------------")

        self.logger.info(f"Accuracy: {accuracy:.4f}")
        self.logger.info(f'Precision: {precision:.4f}')
        self.logger.info(f'Recall: {recall:.4f}')
        self.logger.info(f'F1 Score: {f1:.4f}')
        self.logger.info(f'Brier Score: {brier:.4f}')

    def visualization_best3(self):
        # Load the model weights
        self.load_weights('./batch_results/UFPR0510000/siamese/UFPR05_1/best.weights.h5')
        images = list(self.test_dataset.take(128))  # Pegue 18 pares para um total de 36 imagens
        true_labels = []
        predicted_labels = []
        self.images_1 = images
        fig = plt.figure(figsize=(32, 64))  # Ajuste o tamanho da figura conforme necessário
        rows = 32  # Quatro linhas
        columns = 8  # Nove colunas

        for i in range(128):
            img = images[i][0]  # Assumindo que os dados são uma tupla (imagem, rótulo)
            img = np.squeeze(img)  # Remove quaisquer dimensões extras, se necessário

            imagem1 = img[0]
            imagem2 = img[1]
            label = images[i][1]

            # Fazer previsões para o par de imagens
            imagem1_batch = np.expand_dims(imagem1, axis=0)
            imagem2_batch = np.expand_dims(imagem2, axis=0)

            predicted_label = self.predict([imagem1_batch, imagem2_batch])[0]
            
            true_labels.append(label)
            if predicted_label > 0.5:
                predicted_labels.append(1.0)
            else: 
                predicted_labels.append(0.0)


            
            # Plotar as imagens
            fig.add_subplot(rows, columns, 2*i+1)
            plt.imshow(imagem1, cmap='gray')
            plt.axis('off')

            fig.add_subplot(rows, columns, 2*i+2)
            plt.imshow(imagem2, cmap='gray')
            plt.axis('off')

            plt.title(f'P:{predicted_label.tolist()}, T:{label}, N: {predicted_labels[i]},')

        plt.tight_layout()

        # Salvar a figura
        plt.savefig(f'{self.model_path}/final_prediction_128_ufpr05_10000.png')
        plt.close()
        
        # Calcular métricas de avaliação
        true_labels = np.array(true_labels)
        predicted_labels = np.array(predicted_labels)

        # Converter previsões para classes binárias com um limiar de 0.5
        binary_predictions = predicted_labels.astype(np.float32)

        accuracy = accuracy_score(true_labels, binary_predictions)
        precision = precision_score(true_labels, binary_predictions)
        recall = recall_score(true_labels, binary_predictions)
        f1 = f1_score(true_labels, binary_predictions)
        brier = brier_score_loss(true_labels, predicted_labels)

        print(f'Accuracy: {accuracy:.4f}')
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'F1 Score: {f1:.4f}')
        print(f'Brier Score: {brier:.4f}')
        self.logger.info(f"---------------------- REDE TREINADA COM 10000 IMGENS---------------------------------------")

        self.logger.info(f"Accuracy: {accuracy:.4f}")
        self.logger.info(f'Precision: {precision:.4f}')
        self.logger.info(f'Recall: {recall:.4f}')
        self.logger.info(f'F1 Score: {f1:.4f}')
        self.logger.info(f'Brier Score: {brier:.4f}')



    def visualization(self):
        # Coletar pares de imagens e rótulos reais
        images = list(self.test_dataset.take(64))  # Pegue 18 pares para um total de 36 imagens
        true_labels = []
        predicted_labels = []

        fig = plt.figure(figsize=(32, 16))  # Ajuste o tamanho da figura conforme necessário

        rows = 4  # Quatro linhas
        columns = 9  # Nove colunas

        for i in range(18):
            img = images[i][0]  # Assumindo que os dados são uma tupla (imagem, rótulo)
            img = np.squeeze(img)  # Remove quaisquer dimensões extras, se necessário

            imagem1 = img[0]
            imagem2 = img[1]
            label = images[i][1]

            # Fazer previsões para o par de imagens
            imagem1_batch = np.expand_dims(imagem1, axis=0)
            imagem2_batch = np.expand_dims(imagem2, axis=0)

            predicted_label = self.predict([imagem1_batch, imagem2_batch])[0]
            
            true_labels.append(label)
            if predicted_label > 0.5:
                predicted_labels.append(1.0)
            else: 
                predicted_labels.append(0.0)


            
            # Plotar as imagens
            fig.add_subplot(rows, columns, 2*i+1)
            plt.imshow(imagem1, cmap='gray')
            plt.axis('off')

            fig.add_subplot(rows, columns, 2*i+2)
            plt.imshow(imagem2, cmap='gray')
            plt.axis('off')

            plt.title(f'P:{predicted_label.tolist()}, T:{label}, N: {predicted_labels[i]},')

        plt.tight_layout()

        # Salvar a figura
        plt.savefig(f'{self.model_path}/prediction_64_ufpr05_5000.png')
        plt.close()
        
        # Calcular métricas de avaliação
        true_labels = np.array(true_labels)
        predicted_labels = np.array(predicted_labels)

        # Converter previsões para classes binárias com um limiar de 0.5
        binary_predictions = predicted_labels.astype(np.float32)

        accuracy = accuracy_score(true_labels, binary_predictions)
        precision = precision_score(true_labels, binary_predictions)
        recall = recall_score(true_labels, binary_predictions)
        f1 = f1_score(true_labels, binary_predictions)
        brier = brier_score_loss(true_labels, predicted_labels)

        print(f'Accuracy: {accuracy:.4f}')
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'F1 Score: {f1:.4f}')
        print(f'Brier Score: {brier:.4f}')

        self.logger.info(f"Accuracy: {accuracy:.4f}")
        self.logger.info(f'Precision: {precision:.4f}')
        self.logger.info(f'Recall: {recall:.4f}')
        self.logger.info(f'F1 Score: {f1:.4f}')
        self.logger.info(f'Brier Score: {brier:.4f}')

    def visualization_best4(self):
        # Load the model weights
        self.load_weights('./batch_results/UFPR05 5000/siamese/UFPR05_1/best.weights.h5')
        images = list(self.test_dataset.take(64))  # Pegue 18 pares para um total de 36 imagens
        true_labels = []
        predicted_labels = []

        fig = plt.figure(figsize=(32, 16))  # Ajuste o tamanho da figura conforme necessário

        rows = 4  # Quatro linhas
        columns = 9  # Nove colunas

        for i in range(18):
            img = images[i][0]  # Assumindo que os dados são uma tupla (imagem, rótulo)
            img = np.squeeze(img)  # Remove quaisquer dimensões extras, se necessário

            imagem1 = img[0]
            imagem2 = img[1]
            label = images[i][1]

            # Fazer previsões para o par de imagens
            imagem1_batch = np.expand_dims(imagem1, axis=0)
            imagem2_batch = np.expand_dims(imagem2, axis=0)

            predicted_label = self.predict([imagem1_batch, imagem2_batch])[0]
            
            true_labels.append(label)
            if predicted_label > 0.5:
                predicted_labels.append(1.0)
            else: 
                predicted_labels.append(0.0)


            
            # Plotar as imagens
            fig.add_subplot(rows, columns, 2*i+1)
            plt.imshow(imagem1, cmap='gray')
            plt.axis('off')

            fig.add_subplot(rows, columns, 2*i+2)
            plt.imshow(imagem2, cmap='gray')
            plt.axis('off')

            plt.title(f'P:{predicted_label.tolist()}, T:{label}, N: {predicted_labels[i]},')

        plt.tight_layout()

        # Salvar a figura
        plt.savefig(f'{self.model_path}/final_prediction_64_ufpr05_5000.png')
        plt.close()
        
        # Calcular métricas de avaliação
        true_labels = np.array(true_labels)
        predicted_labels = np.array(predicted_labels)

        # Converter previsões para classes binárias com um limiar de 0.5
        binary_predictions = predicted_labels.astype(np.float32)

        accuracy = accuracy_score(true_labels, binary_predictions)
        precision = precision_score(true_labels, binary_predictions)
        recall = recall_score(true_labels, binary_predictions)
        f1 = f1_score(true_labels, binary_predictions)
        brier = brier_score_loss(true_labels, predicted_labels)

        print(f'Accuracy: {accuracy:.4f}')
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'F1 Score: {f1:.4f}')
        print(f'Brier Score: {brier:.4f}')

        self.logger.info(f"Accuracy: {accuracy:.4f}")
        self.logger.info(f'Precision: {precision:.4f}')
        self.logger.info(f'Recall: {recall:.4f}')
        self.logger.info(f'F1 Score: {f1:.4f}')
        self.logger.info(f'Brier Score: {brier:.4f}')

    def visualization_pairs(self):
        pairs = [self.reference_pairs_train]

        for pair in pairs:
            fig = plt.figure(figsize=(32, 32))
            rows = 2
            columns = 5
            for i in range(0,6,1):
                image = pair[i][0]
                #pdb.set_trace()
                image = tf.keras.utils.load_img(image)
                image = image.resize((105, 105))
                image = tf.keras.utils.img_to_array(image)/255.0
                label = pair[i][1]
                fig.add_subplot(rows, columns, i+1)
                plt.imshow(image)
                plt.axis("off")
                plt.title(f'{label}')
            plt.savefig(f'{self.model_path}/pairs_ref.png')
            plt.close()

    def pairs_visualization(self):
        images = list(self.test_dataset.take(50))  # Pegue 50 pares (100 imagens)
        fig = plt.figure(figsize=(32, 32))  # Ajuste o tamanho da figura conforme necessário
        print(f'Total de pares: {len(images)}')
        rows = 10  # Número de linhas para acomodar 50 pares
        columns = 10  # Número de colunas para acomodar as imagens

        for i in range(50):
            img = images[i][0]  
            img = np.squeeze(img) 

            imagem1 = img[0]
            imagem2 = img[1]
            label = images[i][1]

            fig.add_subplot(rows, columns, 2 * i + 1)
            plt.imshow(imagem1, cmap='gray')
            plt.axis("off")

            fig.add_subplot(rows, columns, 2 * i + 2)  
            plt.imshow(imagem2, cmap='gray')
            plt.axis("off")

            plt.title(f'{label}')

        plt.savefig(f'{self.model_path}/pairs_print.png')
        plt.close()


    def filter_training_dataset(self, name, num, random):
        data = self.dataset[name]
        idx = np.arange(len(data))
        if random:
            np.random.shuffle(idx)
        empty_images = [img for img in data if img[1] == 'Empty']
        occupied_images = [img for img in data if img[1] == 'Occupied']

        if len(empty_images) < num or len(occupied_images) < num:
            raise ValueError("O dataset não possui imagens suficientes para criar os pares necessários.")

        np.random.shuffle(empty_images)
        np.random.shuffle(occupied_images)

        # Seleciona 5000 imagens de cada classe
        empty_images = empty_images[:num]
        occupied_images = occupied_images[:num]

        # Atualiza o dataset com as imagens selecionadas
        self.dataset[name] = empty_images + occupied_images

    def count_images_per_class(self):
        counts = {'train': {'Empty': 0, 'Occupied': 0},
                    'test': {'Empty': 0, 'Occupied': 0},
                    'validation': {'Empty': 0, 'Occupied': 0}}
        
        for datatype in ['train', 'test', 'validation']:
            for img in self.dataset[datatype]:
                counts[datatype][img[1]] += 1
        
        return counts
