import tensorflow as tf

def get_data(val_size):
    mnist = tf.keras.datasets.mnist
    (x_train_full, y_train_full), (x_test, y_test) = mnist.load_data()

    x_val, x_train = x_train_full[:val_size] / 255., x_train_full[val_size:]/255.
    y_val, y_train = y_train_full[:val_size], y_train_full[val_size:]

    x_test = x_test/255.

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)