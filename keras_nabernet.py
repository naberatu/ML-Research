
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten

# ================================================================
def nabernet(n_classes=2, im_size=(224, 224), n_channels=3):
    # Model Parameters
    cdims = [32, 64, 64, 64, 64]
    fdims = [64, 32, 1 if n_classes <= 2 else n_classes]
    # fdims = [100, 64, n_classes]

    inputs = Input(shape=im_size + (n_channels, ))
    x = None
    for i, dim in enumerate(cdims):
        x = Conv2D(dim, 3, activation='relu')(inputs) if i == 0 else Conv2D(dim, 3, activation='relu')(x)
        if i % 2 != 0:
            x = MaxPooling2D(2, 2)(x)

    x = Flatten()(x)    # The bridge between Conv & Dense (FC) layers
    for i, dim in enumerate(fdims):
        if i < len(fdims) - 1:
            x = Dense(dim, activation='relu')(x)
    outputs = Dense(fdims[-1], activation='sigmoid' if n_classes == 2 else 'softmax')(x)
    # outputs = Dense(fdims[-1], activation='softmax')(x)

    return Model(inputs, outputs)
# ================================================================
