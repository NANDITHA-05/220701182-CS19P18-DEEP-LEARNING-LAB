import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

# Load and prepare data
(x_train, _), (_, _) = mnist.load_data()
x_train = (x_train.astype("float32") - 127.5) / 127.5  # [-1,1]
x_train = x_train.reshape(-1, 28*28)

buffer_size = x_train.shape[0]
batch_size = 256

train_dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(buffer_size).batch(batch_size)

latent_dim = 100

# Generator
def build_generator():
    model = tf.keras.Sequential([
        layers.Dense(256, activation="relu", input_shape=(latent_dim,)),
        layers.Dense(512, activation="relu"),
        layers.Dense(28*28, activation="tanh")
    ])
    return model

# Discriminator
def build_discriminator():
    model = tf.keras.Sequential([
        layers.Dense(512, activation="relu", input_shape=(28*28,)),
        layers.Dense(256, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])
    return model

generator = build_generator()
discriminator = build_discriminator()

cross_entropy = tf.keras.losses.BinaryCrossentropy()

gen_optimizer = tf.keras.optimizers.Adam(1e-4)
disc_optimizer = tf.keras.optimizers.Adam(1e-4)

# Training step
@tf.function
def train_step(real_images):
    noise = tf.random.normal([batch_size, latent_dim])

    # Train discriminator
    with tf.GradientTape() as disc_tape:
        generated = generator(noise, training=True)

        real_output = discriminator(real_images, training=True)
        fake_output = discriminator(generated, training=True)

        real_loss = cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
        disc_loss = real_loss + fake_loss

    grads_disc = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    disc_optimizer.apply_gradients(zip(grads_disc, discriminator.trainable_variables))

    # Train generator
    noise = tf.random.normal([batch_size, latent_dim])
    with tf.GradientTape() as gen_tape:
        generated = generator(noise, training=True)
        fake_output = discriminator(generated, training=True)
        gen_loss = cross_entropy(tf.ones_like(fake_output), fake_output)

    grads_gen = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gen_optimizer.apply_gradients(zip(grads_gen, generator.trainable_variables))

# Training loop (few epochs for demo)
epochs = 20
for epoch in range(epochs):
    for batch in train_dataset:
        train_step(batch)

    print(f"Epoch {epoch+1}/{epochs} done")

# Generate and show some fake digits
noise = tf.random.normal([16, latent_dim])
generated_images = generator(noise, training=False).numpy().reshape(-1,28,28)

plt.figure(figsize=(6,6))
for i in range(16):
    plt.subplot(4,4,i+1)
    plt.imshow((generated_images[i] + 1)/2, cmap='gray')  # back to [0,1]
    plt.axis('off')
plt.tight_layout()
plt.show()
