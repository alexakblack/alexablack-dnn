import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.activations as activations
import tensorflow.keras.optimizers as optimizers
import tensorflow.keras.losses as losses

# Load Data Here

inputs = tf.keras.Input(shape = (227, 227, 3))
# First layer
outputs = layers.Conv2D(
    12, 
    11, 
    strides=4, 
    activation=activations.relu,
)(inputs)
outputs = layers.MaxPool2D(pool_size=3, strides=2)(outputs)
...

loss = losses.CategoricalCrossEntropy()
optimizer = optimizers.Adam(learning_rate = 0.0001)
model = tf.keras.Model(inputs, outputs)
model.compile(
    loss = loss
    optimizer = optimizer
    metrics = ['accuracy']
)

model.summary()