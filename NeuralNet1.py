import tensorflow as tf

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# normalise the data
x_train = x_train/255.0
x_test = x_test/255.0

#instantiate model
model = tf.keras.models.Sequential([
    #tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28)),
    #tf.keras.layers.MaxPool2D((2, 2)),
    #tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
    #tf.keras.layers.MaxPool2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') > 0.99:
            print("\nReached 99% accuracy so cancelling training")
            self.model.stop_training = True

callbacks = myCallback()

model.fit(x_train, y_train, epochs=9, callbacks=[callbacks])

# predict on test set
model.evaluate(x_test, y_test)
