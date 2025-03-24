from tensorflow import keras

def static():
    base_model = keras.applications.MobileNetV3Large(
                    input_shape=(128,128,3),
                    include_top=False)
    base_model.trainable = False

    model = keras.Sequential(
        layers=[
            keras.layers.Rescaling(1./255),
            base_model,
            keras.layers.Flatten(),
            keras.layers.Dense(1024,activation='relu'),
            keras.layers.Dense(128,activation='relu'),
            keras.layers.Dense(2,activation='softmax')
        ],
        name='mobilenet_static')

    model.compile(optimizer='Adam',
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    return model

def dynamic():
    base_model = keras.applications.MobileNetV3Large(
                    input_shape=(128,128,3),
                    include_top=False)
    base_model.trainable = True

    model = keras.Sequential(
        layers=[
            keras.layers.Rescaling(1./255),
            base_model,
            keras.layers.Flatten(),
            keras.layers.Dense(1024,activation='relu'),
            keras.layers.Dense(128,activation='relu'),
            keras.layers.Dense(2,activation='softmax')
        ],
        name='mobilenet_dynamic')

    model.compile(optimizer='Adam',
                    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])

    return model
