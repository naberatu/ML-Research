
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense


# ================================================================
def nabernet(n_classes=2, im_size=(224, 224), n_channels=3):
    # Model Parameters
    cdims = [32, 48, 64, 64, 64]
    fdims = [100, 64, n_classes]

    # Model initialization
    model = Sequential()

    # Convolutional layers
    for i, dim in enumerate(cdims):
        if i == 0:
            model.add(Conv2D(cdims[i], kernel_size=3, input_shape=(im_size[0], im_size[1], n_channels), activation='relu'))
        else:
            model.add(Conv2D(cdims[i], kernel_size=3, activation='relu'))
        if i % 2 != 0:
            model.add(MaxPooling2D(2, 2))

    # Fully-connected layers
    for dim in fdims:
        if dim == n_classes:
            model.add(Dense(dim, activation='softmax'))
        else:
            model.add(Dense(dim, activation='relu'))

    return model
# ================================================================
