import tensorflow as tf
from tensorflow import keras

# Define some parameters
sequence_length = 10  # Length of input sequences
num_features = 4  # Number of features in each element

# Create some sample data
X = tf.random.uniform((100, sequence_length, num_features))  # Input data

# Build the RNN model
model = keras.Sequential([
  keras.layers.SimpleRNN(units=8, return_sequences=True),  # RNN layer with 8 units
  keras.layers.SimpleRNN(units=4),  # Another RNN layer with 4 units
  keras.layers.Dense(1)  # Output layer with 1 unit
])

# Compile the model
model.compile(loss="mse", optimizer="adam")

# Generate some target data (can be replaced with your actual target)
y = X[:, -1, :]  # Simulate predicting the last element

# Train the model
model.fit(X, y, epochs=10, batch_size=32)

# Make predictions on new data
new_data = tf.random.uniform((1, sequence_length, num_features))
prediction = model.predict(new_data)

# Print the prediction
print(prediction)
print("input X : ", X[0])