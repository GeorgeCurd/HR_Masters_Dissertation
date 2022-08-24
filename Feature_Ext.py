from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LeakyReLU, BatchNormalization
from tensorflow.keras.utils import plot_model
from matplotlib import pyplot
import numpy as np
from Create_Preds_Runner import X_test_norm, X_train_norm_ext
from keras import Sequential
from keras.regularizers import l1
from tensorflow.keras.models import Model, load_model

# 2 Layer
# define encoder
n_inputs = X_train_norm_ext.shape[1]
visible = Input(shape=(n_inputs,))
# encoder level 1
e = Dense(n_inputs/5)(visible)
e = BatchNormalization()(e)
e = LeakyReLU()(e)
# encoder level 2
e = Dense(n_inputs/25)(e)
e = BatchNormalization()(e)
e = LeakyReLU()(e)
# bottleneck
n_bottleneck = round(float(n_inputs) / 150.0)

# n_bottleneck = n_inputs
bottleneck = Dense(n_bottleneck)(e)

# define decoder, level 1
d = Dense(n_inputs/25)(bottleneck)
d = BatchNormalization()(d)
d = LeakyReLU()(d)
# define decoder, level 2
d = Dense(n_inputs/5)(d)
d = BatchNormalization()(d)
d = LeakyReLU()(d)
# output layer
output = Dense(n_inputs, activation='sigmoid')(d)

# define autoencoder model
model = Model(inputs=visible, outputs=output)

# compile autoencoder model
model.compile(optimizer='adam', loss='mse')

# fit the autoencoder model to reconstruct input
hist2 = model.fit(X_train_norm_ext, X_train_norm_ext, epochs=100, batch_size=128, verbose=1,
                 validation_data=(X_test_norm, X_test_norm))

# 1 Layer
# define encoder
n_inputs = X_train_norm_ext.shape[1]
visible = Input(shape=(n_inputs,))
# encoder level 1
e = Dense(n_inputs/12)(visible)
e = BatchNormalization()(e)
e = LeakyReLU()(e)
# bottleneck
n_bottleneck = round(float(n_inputs) / 150.0)

# n_bottleneck = n_inputs
bottleneck = Dense(n_bottleneck)(e)

# define decoder, level 1
d = Dense(n_inputs/12)(bottleneck)
d = BatchNormalization()(d)
d = LeakyReLU()(d)
# output layer
output = Dense(n_inputs, activation='sigmoid')(d)

# define autoencoder model
model = Model(inputs=visible, outputs=output)

# compile autoencoder model
model.compile(optimizer='adam', loss='mse')

# fit the autoencoder model to reconstruct input
hist = model.fit(X_train_norm_ext, X_train_norm_ext, epochs=100, batch_size=128, verbose=1,
                 validation_data=(X_test_norm, X_test_norm))

# 3 Layer
# define encoder
n_inputs = X_train_norm_ext.shape[1]
visible = Input(shape=(n_inputs,))
# encoder level 1
e = Dense(n_inputs/3)(visible)
e = BatchNormalization()(e)
e = LeakyReLU()(e)
# encoder level 2
e = Dense(n_inputs/9)(e)
e = BatchNormalization()(e)
e = LeakyReLU()(e)
# encoder level 3
e = Dense(n_inputs/27)(e)
e = BatchNormalization()(e)
e = LeakyReLU()(e)
# bottleneck
n_bottleneck = round(float(n_inputs) / 150.0)

# n_bottleneck = n_inputs
bottleneck = Dense(n_bottleneck)(e)

# define decoder, level 1
d = Dense(n_inputs/27)(bottleneck)
d = BatchNormalization()(d)
d = LeakyReLU()(d)
# define decoder, level 2
d = Dense(n_inputs/9)(d)
d = BatchNormalization()(d)
d = LeakyReLU()(d)
# define decoder, level 3
d = Dense(n_inputs/3)(d)
d = BatchNormalization()(d)
d = LeakyReLU()(d)
# output layer
output = Dense(n_inputs, activation='sigmoid')(d)

# define autoencoder model
model = Model(inputs=visible, outputs=output)

# compile autoencoder model
model.compile(optimizer='adam', loss='mse')

# fit the autoencoder model to reconstruct input
hist3 = model.fit(X_train_norm_ext, X_train_norm_ext, epochs=100, batch_size=128, verbose=1,
                 validation_data=(X_test_norm, X_test_norm))


# plot loss
pyplot.plot(hist.history['val_loss'], label='1-Layer')
pyplot.plot(hist2.history['val_loss'], label='2-Layer')
pyplot.plot(hist3.history['val_loss'], label='3-Layer')
limits = [0, 100, 0.70, 0.9]
pyplot.axis(limits)
pyplot.legend()
pyplot.xlabel("Epochs")
pyplot.ylabel(" Validation Loss (MSE)")
pyplot.show()

# Find best loss and Val Loss Results
loss = hist3.history['val_loss']
best_loss = min(loss)
bl_index = loss.index(best_loss)
print(best_loss)
print(bl_index)

# # Define the encoder model (without decoder)
# encoder = Model(inputs=visible, outputs=bottleneck)
# # # save the encoder to file
# encoder.save('encoder.h5')
