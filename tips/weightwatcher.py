import os

import numpy as np

os.environ["KERAS_BACKEND"] = "tensorflow"

import keras


def conv2D_Wmats(weight_tensor):
    """Convert a Keras conv2d layer into a list of 2D weight matrices.

    Special case of weightwatchers.py:WWLayer.conv2D_Wmats for this demo.
    """
    weight_matrices = []
    s = weight_tensor.shape
    imax, jmax, N, M = s[0], s[1], s[2], s[3]
    print(f"{imax=}, {jmax=}, {N=}, {M=}")

    for i in range(imax):
        for j in range(jmax):
            W = weight_tensor[i, j, :, :]
            if W.shape[1] < W.shape[0]:
                N, M = M, N
                W = W.T
            weight_matrices.append(W)

    receptive_field_size = imax * jmax
    return weight_matrices, N, M, receptive_field_size


def mnist():
    """Train an MNIST model and save it to disk.

    Code taken directly from Keras tutorial: https://keras.io/getting_started/intro_to_keras_for_engineers/
    """
    # Load the data and split it between train and test sets
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Scale images to the [0, 1] range
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    # Make sure images have shape (28, 28, 1)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    print("x_train shape:", x_train.shape)
    print("y_train shape:", y_train.shape)
    print(x_train.shape[0], "train samples")
    print(x_test.shape[0], "test samples")

    # Model parameters
    num_classes = 10
    input_shape = (28, 28, 1)

    model = keras.Sequential(
        [
            keras.layers.Input(shape=input_shape),
            keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Conv2D(128, kernel_size=(3, 3), activation="relu"),
            keras.layers.Conv2D(128, kernel_size=(3, 3), activation="relu"),
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(num_classes, activation="softmax"),
        ],
    )
    model.summary()

    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(),
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="acc"),
        ],
    )

    batch_size = 128
    epochs = 20

    callbacks = [
        keras.callbacks.ModelCheckpoint(filepath="model_at_epoch_{epoch}.keras"),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=2),
    ]
    model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.15,
        callbacks=callbacks,
    )
    score = model.evaluate(x_test, y_test, verbose=0)
    print(f"{score=}")
    model.save("data/trained_mnist.keras")
    return model


if __name__ == "__main__":
    # uncomment to retrain
    # mnist()

    model = keras.saving.load_model("data/trained_mnist.keras")
