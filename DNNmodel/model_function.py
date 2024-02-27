import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.activations as activations
import tensorflow.keras.optimizers as optimizers
import tensorflow.keras.losses as losses

# Import Data Here


input_size = (227, 227, 3)
model = tf.keras.Sequential()
# Input: 227 x 227 x 3
model.add(layers.Conv2D(
    12, 
    11, 
    strides=4, 
    activation=activations.relu,
    input_shape=input_size,
    ))
# Size: 55 x 55 x 12
model.add(layers.MaxPool2D(
    pool_size=3,
    strides=2,
))
# Size: 27 x 27 x 12
model.add(layers.Conv2D(
    18,
    3,
    strides=1,
    activation=activations.relu,
))
#Size: 25 x 25 x 18
model.add(layers.MaxPool2D(
    pool_size=3,
    strides=2,
))
# Size: 12 x 12 x 18
model.add(layers.Flatten())
# Size: 2592
model.add(layers.Dense(512, activation=activations.relu))
model.add(layers.Dense(128, activation=activations.relu))
model.add(layers.Dense(32, activation=activations.relu))
# Size of last Dense layer MUST match # of classes
model.add(layers.Dense(5, activation=activations.softmax))
optimizer = optimizers.Adam(learning_rate=0.0001)
model.loss = losses.CategoricalCrossentropy
model.compile(
    loss = losses,
    optimizer = optimizer,
    metrics = ['accuracy'],
)

model.summary()