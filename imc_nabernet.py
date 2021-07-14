
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten

# ================================================================
def nabernet(n_classes=2, im_size=(224, 224), n_channels=3):
    # Model Parameters
    cdims = [64, 64, 64, 64, 64]
    fdims = [128, 64, 1 if n_classes <= 2 else n_classes]

    inputs = Input(shape=im_size + (n_channels, ))

    x = Conv2D(cdims[0], 3, activation='relu')(inputs)
    x = Conv2D(cdims[1], 3, activation='relu')(x)
    x = MaxPooling2D(2, 2)(x)
    x = Conv2D(cdims[2], 3, activation='relu')(x)
    x = Conv2D(cdims[3], 3, activation='relu')(x)
    x = MaxPooling2D(2, 2)(x)
    x = Conv2D(cdims[4], 3, activation='relu')(x)
    x = Flatten()(x)
    x = Dense(fdims[0], activation='relu')(x)
    x = Dense(fdims[1], activation='relu')(x)
    outputs = Dense(fdims[2], activation='sigmoid')(x)

    return Model(inputs, outputs)


    # # Entry block
    # x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    # x = layers.BatchNormalization()(x)
    # x = layers.Activation("relu")(x)
    #
    # x = layers.Conv2D(64, 3, padding="same")(x)
    # x = layers.BatchNormalization()(x)
    # x = layers.Activation("relu")(x)
    #
    # previous_block_activation = x  # Set aside residual
    #
    # for size in [128, 256, 512, 728]:
    #     x = layers.Activation("relu")(x)
    #     x = layers.SeparableConv2D(size, 3, padding="same")(x)
    #     x = layers.BatchNormalization()(x)
    #
    #     x = layers.Activation("relu")(x)
    #     x = layers.SeparableConv2D(size, 3, padding="same")(x)
    #     x = layers.BatchNormalization()(x)
    #
    #     x = layers.MaxPooling2D(3, strides=2, padding="same")(x)
    #
    #     # Project residual
    #     residual = layers.Conv2D(size, 1, strides=2, padding="same")(
    #         previous_block_activation
    #     )
    #     x = layers.add([x, residual])  # Add back residual
    #     previous_block_activation = x  # Set aside next residual
    #
    # x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    # x = layers.BatchNormalization()(x)
    # x = layers.Activation("relu")(x)
    #
    # x = layers.GlobalAveragePooling2D()(x)
    # if n_classes == 2:
    #     activation = "sigmoid"
    #     units = 1
    # else:
    #     activation = "softmax"
    #     units = n_classes
    #
    # x = layers.Dropout(0.5)(x)
    # outputs = layers.Dense(units, activation=activation)(x)
    # return Model(inputs, outputs)



    # # Model Parameters
    # cdims = [32, 48, 64, 64, 64]
    # fdims = [100, 64, 1 if n_classes <= 2 else n_classes]
    #
    # inputs = Input(shape=im_size + (n_channels,))
    # x = None
    # for i, dim in enumerate(cdims):
    #     x = Conv2D(dim, 3)(inputs) if i == 0 else Conv2D(dim, 3)(x)
    #     x = Activation('relu')(x)
    #     if i % 2 != 0:
    #         x = MaxPooling2D(2, 2)(x) if i % 2 != 0 else \
    #
    # for i, dim in enumerate(fdims):
    #     if i < len(fdims) - 1:
    #         x = Dense(dim, activation='relu')(x)
    #
    # outputs = Dense(fdims[-1], activation='sigmoid' if n_classes == 2 else 'softmax')(x)
    # return Model(inputs, outputs)


    # # Model initialization
    # model = Sequential()
    #
    # # Convolutional layers
    # for i, dim in enumerate(cdims):
    #     if i == 0:
    #         model.add(Conv2D(cdims[i], kernel_size=3, input_shape=(im_size[0], im_size[1], n_channels), activation='relu'))
    #     else:
    #         model.add(Conv2D(cdims[i], kernel_size=3, activation='relu'))
    #     if i % 2 != 0:
    #         model.add(MaxPooling2D(2, 2))
    #
    # # Fully-connected layers
    # for dim in fdims:
    #     if dim == n_classes:
    #         model.add(Dense(dim, activation='sigmoid'))
    #     else:
    #         model.add(Dense(dim, activation='relu'))

    # return model
# ================================================================
