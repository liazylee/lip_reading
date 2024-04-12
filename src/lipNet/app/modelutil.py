import os

from keras.src.applications import ResNet50
from tensorflow.keras.layers import Conv3D, LSTM, Dense, Dropout, Bidirectional, MaxPool3D, Activation, TimeDistributed, \
    Flatten
from tensorflow.keras.models import Sequential

script_dir = os.path.dirname(os.path.abspath(__file__))


def load_model() -> Sequential:
    base_model = ResNet50(weights=None, include_top=False, input_shape=(70, 140, 1))

    # If you wish to Freeze layers
    for layer in base_model.layers:
        layer.trainable = True

    model = Sequential()
    model.add(base_model)
    model = Sequential()
    # Add a 3D convolutional layer with 128 filters, a kernel size of (3, 3, 3), and input shape of (75, 46, 140, 1)
    # The 'padding' parameter is set to 'same' to ensure that the spatial dimensions of the output remain the same as the input.
    model.add(Conv3D(128, 3, input_shape=(75, 70, 140, 1), padding='same'))
    # Add a ReLU activation function to introduce non-linearity after the convolutional layer.
    model.add(Activation('relu'))
    # Add a 3D max pooling layer with a pool size of (1, 2, 2)
    # This layer reduces the spatial dimensions by taking the maximum value over a 1x2x2 region.
    model.add(MaxPool3D((1, 2, 2)))

    model.add(Conv3D(256, 3, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool3D((1, 2, 2)))

    model.add(Conv3D(75, 3, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool3D((1, 2, 2)))
    # Add a dropout layer with a dropout rate of 0.5 after the convolutional layers.
    model.add(TimeDistributed(Flatten()))

    model.add(Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)))
    model.add(Dropout(.5))

    model.add(Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)))
    model.add(Dropout(.5))

    model.add(Dense(29, kernel_initializer='he_normal', activation='softmax'))

    model.load_weights(os.path.join('..', 'models', 'lip_reading_baseline_140*70.h5'))

    return model
