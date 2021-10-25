import tensorflow as tf
import numpy as np
import os
import time

def get_timestamp(name):
    timestamp = time.asctime().replace(" ", "_").replace(":", "_")
    return f'{name}_at_{timestamp}'

def get_callbacks(config, x_train):
    logs = config["logs"]
    unique_dir_name = get_timestamp("tb_logs")
    TENSORBOARD_ROOT_LOG_DIR = os.path.join(logs["logs_dir"], logs["TENSORBOARD_ROOT_LOG_DIR"], unique_dir_name)

    os.makedirs(TENSORBOARD_ROOT_LOG_DIR, exist_ok=True)

    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=TENSORBOARD_ROOT_LOG_DIR)
    file_writer = tf.summary.create_file_writer(logdir=TENSORBOARD_ROOT_LOG_DIR)
    with file_writer.as_default():
        images = np.reshape(x_train[10:30], (-1, 28, 28, 1)) #(20, 28, 28, 1)
        tf.summary.image("20 handwritten images", images, max_outputs=25, step=0)


    params = config["params"]
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=params["patience"], restore_best_weights=params["restore_best_weights"])


    artifacts = config["artifacts"]
    CHKPT_DIR = os.path.join(artifacts["artifacts_dir"], artifacts["CHECKPOINT_DIR"])

    os.makedirs(CHKPT_DIR, exist_ok=True)

    chkpt_path = os.path.join(CHKPT_DIR, 'model_chpt.h5')

    checkpointing_cb = tf.keras.callbacks.ModelCheckpoint(chkpt_path, save_best_only=True)

    return [tensorboard_cb, early_stopping_cb, checkpointing_cb]