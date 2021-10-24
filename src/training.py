from utils.data_mgt import get_data
from utils.common import read_config
from utils.model import create_model, save_model
from utils.callbacks import get_callbacks

import os
import argparse

def training(config_path):
    config = read_config(config_path)
    
    val_size = config["params"]["validation_datasize"]
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = get_data(val_size)

    LOSS_FUNCTION = config["params"]["loss_function"]
    OPTIMIZER = config["params"]["optimizer"]
    METRICS = config["params"]["metrics"]
    NUM_CLASSES = config["params"]["num_classes"]

    model = create_model(LOSS_FUNCTION, OPTIMIZER, METRICS, NUM_CLASSES)

    EPOCHS = config["params"]["epochs"]
    VALIDATION = (x_val, y_val)


    COLLBACK_LIST = get_callbacks(config, x_train)

    history = model.fit(x_train, y_train, epochs=EPOCHS, validation_data=VALIDATION, callbacks=COLLBACK_LIST)

    artifacts_dir = config["artifacts"]["artifacts_dir"]
    model_dir = config["artifacts"]["model_dir"]
    
    model_dir_path = os.path.join(artifacts_dir, model_dir)
    os.makedirs(model_dir_path, exist_ok=True)
    
    model_name = config["artifacts"]["model_name"]

    save_model(model, model_name, model_dir_path)
if __name__ == '__main__':
    args = argparse.ArgumentParser()

    args.add_argument("--config", "-c", default="config.yml")

    parsed_args = args.parse_args()

    training(config_path=parsed_args.config)