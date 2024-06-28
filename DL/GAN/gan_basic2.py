import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Dense, Reshape, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist

# Load MNIST dataset
(x_train, _), (_, _) = mnist.load_data()

# Normalize data
x_train = x_train / 255.0

# Define generator model
generator = Sequential([
    Dense(256, input_dim=100, activation='relu'),
    Dense(512, activation='relu'),
    Dense(1024, activation='relu'),
    Dense(784, activation='tanh'),  # Using tanh instead of sigmoid for better distribution
    Reshape((28, 28))
])

# Define discriminator model
discriminator = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(1024, activation='relu'),
    Dense(512, activation='relu'),
    Dense(256, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile discriminator
discriminator.compile(optimizer=Adam(learning_rate=0.0002, beta_1=0.5), loss='binary_crossentropy', metrics=['accuracy'])

# Combined GAN model
discriminator.trainable = False
gan_input = tf.keras.Input(shape=(100,))
x = generator(gan_input)
gan_output = discriminator(x)
gan = tf.keras.Model(gan_input, gan_output)
gan.compile(optimizer=Adam(learning_rate=0.0002, beta_1=0.5), loss='binary_crossentropy')

# Training GAN
def train_gan(epochs, batch_size):
    for epoch in range(epochs):
        for _ in range(len(x_train) // batch_size):
            # Create noise vectors
            noise = np.random.normal(0, 1, (batch_size, 100))
            generated_images = generator.predict(noise)

            # Get a random batch of real images
            real_images = x_train[np.random.randint(0, x_train.shape[0], batch_size)]

            # Create training data for the discriminator
            x_combined = np.concatenate([real_images, generated_images])
            y_combined = np.concatenate([np.ones((batch_size, 1)), np.zeros((batch_size, 1))])

            # Train the discriminator
            discriminator_loss = discriminator.train_on_batch(x_combined, y_combined)

            # Create noise vectors
            noise = np.random.normal(0, 1, (batch_size, 100))
            y_generator = np.ones((batch_size, 1))

            # Train the generator
            generator_loss = gan.train_on_batch(noise, y_generator)

        print(f'Epoch: {epoch+1}, Discriminator Loss: {discriminator_loss[0]}, Generator Loss: {generator_loss}')

# Train GAN
train_gan(epochs=50, batch_size=128)

# Generate images using trained generator
def generate_images(n):
    noise = np.random.normal(0, 1, (n, 100))
    generated_images = generator.predict(noise)
    generated_images = 0.5 * generated_images + 0.5  # Rescale images from [-1, 1] to [0, 1]
    plt.figure(figsize=(10, 10))
    for i in range(n):
        plt.subplot(10, 10, i+1)
        plt.imshow(generated_images[i], cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig("mygraph2.png")
    #plt.show()

# Generate and display generated images
generate_images(100)
