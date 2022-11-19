import tensorflow as tf
print("TensorFlow version:", tf.__version__)



# Load MNIST Dataset
mnist = tf.keras.datasets.mnist

    #Convert data from int to float
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0



# Build a Machine Learning Model
    # Build a keras Sequential model by stacking layers
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])
    # Return a vector of logits or log odds
predictions = model(x_train[:1]).numpy()
predictions

    # Converts logits to probbabilities for each class
tf.nn.softmax(predictions).numpy()

# Define a loss function for training 
    # SparseCategoricalCrossentropy takes a vector of logits, a True index, and returns a scalar loss for each example.
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # This loss is equal to the negative log probability of the true class: The loss is zero if the model is sure of the correct class.
    # This untrained model gives probabilities close to random (1/10 for each class), so the initial loss should be close to -tf.math.log(1/10) ~= 2.3.
loss_fn(y_train[:1], predictions).numpy()



# Configure and compile the model using Keras Model.compile
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])



# Train and Evaluate Model
    # Adjust Model Parameters, Minimize Loss
model.fit(x_train, y_train, epochs=5)

    # Check Model Perfoormance
model.evaluate(x_test,  y_test, verbose=2)

    # Wrap the Trained Model
probability_model = tf.keras.Sequential([
  model,
  tf.keras.layers.Softmax()
])

    # Attaches the softmax to Trained Model
probability_model(x_test[:5])