import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np

# Load data
(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train / 255.0
x_test  = x_test / 255.0

x_train = x_train[..., tf.newaxis]  # (batch,28,28,1)
x_test  = x_test[..., tf.newaxis]

# Encoder
inputs = layers.Input(shape=(28,28,1))
x = layers.Flatten()(inputs)
encoded = layers.Dense(64, activation='relu')(x)

# Decoder
x = layers.Dense(28*28, activation='sigmoid')(encoded)
decoded = layers.Reshape((28,28,1))(x)

autoencoder = models.Model(inputs, decoded)
autoencoder.compile(optimizer='adam', loss='mse')

autoencoder.summary()

autoencoder.fit(x_train, x_train,
                epochs=5,
                batch_size=256,
                validation_split=0.1)

# Show original vs reconstructed images
decoded_imgs = autoencoder.predict(x_test[:10])

plt.figure(figsize=(10,4))
for i in range(10):
    # original
    plt.subplot(2,10,i+1)
    plt.imshow(x_test[i].reshape(28,28), cmap='gray')
    plt.axis('off')
    if i == 0: plt.ylabel("Original")

    # reconstructed
    plt.subplot(2,10,10+i+1)
    plt.imshow(decoded_imgs[i].reshape(28,28), cmap='gray')
    plt.axis('off')
    if i == 0: plt.ylabel("Decoded")
plt.tight_layout()
plt.show()
