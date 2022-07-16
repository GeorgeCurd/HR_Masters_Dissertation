from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LeakyReLU, BatchNormalization
from tensorflow.keras.utils import plot_model
from matplotlib import pyplot
import numpy as np
from Create_Preds_Runner import
from keras import Sequential
from keras.regularizers import l1

# def build_autoencoder_model(df):
# define encoder
n_inputs = df.shape[1]
visible = Input(shape=(n_inputs,))
# encoder level 1
e = Dense(n_inputs * 2)(visible)
e = BatchNormalization()(e)
e = LeakyReLU()(e)
# encoder level 2
e = Dense(n_inputs)(e)
e = BatchNormalization()(e)
e = LeakyReLU()(e)
# bottleneck
n_bottleneck = round(float(n_inputs) / 3.0)
# n_bottleneck = n_inputs
bottleneck = Dense(n_bottleneck)(e)

# define decoder, level 1
d = Dense(n_inputs)(bottleneck)
d = BatchNormalization()(d)
d = LeakyReLU()(d)
# decoder level 2
d = Dense(n_inputs * 2)(d)
d = BatchNormalization()(d)
d = LeakyReLU()(d)
# output layer
output = Dense(n_inputs, activation='sigmoid')(d)

# define autoencoder model
model = Model(inputs=visible, outputs=output)

# compile autoencoder model
model.compile(optimizer='adam', loss='mse')

# fit the autoencoder model to reconstruct input
hist = model.fit(train_normX, train_normX, epochs=25, batch_size=16, verbose=2,
                 validation_data=(test_normX, test_normX))

# plot loss
pyplot.plot(hist.history['loss'], label='train')
pyplot.plot(hist.history['val_loss'], label='test')
limits = [0, 50, 0, .004]
pyplot.axis(limits)
pyplot.legend()
pyplot.xlabel("Epochs")
pyplot.ylabel("MSE")
pyplot.show()

# Define the encoder model (without decoder)
encoder = Model(inputs=visible, outputs=bottleneck)
# save the encoder to file
encoder.save('encoder.h5')

