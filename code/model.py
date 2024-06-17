from tensorflow import keras
from tensorflow.keras import layers
from config import cfg
import optuna
from optuna.trial import TrialState

def cnn_model_optuna(trial):
    input_data = keras.Input(shape=(cfg.row, cfg.column, 1))
    x = layers.Conv2D(32, (3, 3), activation='relu')(input_data)
    x = layers.AveragePooling2D()(x)
    x = layers.Conv2D(64, (1, 1), activation="relu")(x)
    x = layers.AveragePooling2D()(x)
    # x = layers.Flatten()(x)
    x = layers.Dense(32, activation='relu')(x)
    dropout = trial.suggest_float("dropout_{}".format(i), 0.1, 0.5)
    x = layers.Dropout(rate=dropout)(x)
    model_output = layers.Dense(cfg.category, activation='softmax')(x)
    model = keras.Model(input_data, model_output)
    return model

# 基础的CNN模型
def cnn_model():
    input_data = keras.Input(shape=(cfg.row, cfg.column, 1))
    x = layers.Conv2D(32, (3, 3), activation='relu')(input_data)
    x = layers.AveragePooling2D()(x)
    x = layers.Conv2D(64, (1, 1), activation="relu")(x)
    x = layers.AveragePooling2D()(x)
    # x = layers.Flatten()(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    model_output = layers.Dense(cfg.category, activation='softmax')(x)
    model = keras.Model(input_data, model_output)
    return model

    # input_data = keras.Input(shape=(cfg.row, cfg.column, 1))
    # x = layers.Conv2D(128, (3, 3), activation='relu')(input_data)
    # x = layers.AveragePooling2D()(x)
    # x = layers.Conv2D(256, (1, 1), activation="relu")(x)
    # x = layers.AveragePooling2D()(x)
    # x = layers.Flatten()(x)
    # x = layers.Dense(128, activation='relu')(x)
    # x = layers.Dropout(0.2)(x)
    # x = layers.Dense(64, activation='relu')(x)
    # model_output = layers.Dense(cfg.category, activation='softmax')(x)
    # model = keras.Model(input_data, model_output)
    # return model

    # 准确率90%
    # input_data = keras.Input(shape=(cfg.row, cfg.column, 1))
    # x = layers.Conv2D(32, (3, 3), activation='relu')(input_data)
    # x = layers.AveragePooling2D()(x)
    # x = layers.Conv2D(64, (1, 1), activation="relu")(x)
    # x = layers.AveragePooling2D()(x)
    #
    # x = layers.Flatten()(x)
    # x = layers.Dense(64, activation='relu')(x)
    # # x = layers.Dropout(0.5)(x)
    # x = layers.Dense(32, activation='relu')(x)
    # model_output = layers.Dense(cfg.category, activation='softmax')(x)
    # model = keras.Model(input_data, model_output)
    # return model

# def cnn_model():
#     input_data = keras.Input(shape=(cfg.row, cfg.column, 1))
#     x = layers.Conv2D(32, (3, 3), activation='relu', padding="same")(input_data)
#     # x = layers.AveragePooling2D()(x)
#     # x = layers.BatchNormalization()(x)
#     x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
#     # x = layers.AveragePooling2D()(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.Add()([x, input_data])
#
#     x = layers.Flatten()(x)
#     x = layers.Dense(32, activation='relu')(x)
#     x = layers.Dropout(0.2)(x)
#     model_output = layers.Dense(cfg.category, activation='softmax')(x)
#     model = keras.Model(input_data, model_output)

    # input_data = keras.Input(shape=(cfg.row, cfg.column, 1))
    # x = layers.Conv2D(32, (3, 3), activation='relu')(input_data)
    # x = layers.MaxPooling2D()(x)
    # x = layers.Conv2D(64, (1, 1), activation="relu")(x)
    # x = layers.MaxPooling2D()(x)
    #
    # x = layers.Flatten()(x)
    # x = layers.Dense(32, activation='relu')(x)
    # x = layers.Dropout(0.2)(x)
    # model_output = layers.Dense(cfg.category, activation='softmax')(x)
    # model = keras.Model(input_data, model_output)
    # return model

    # input_data = keras.Input(shape=(625, 8, 1))
    # x = layers.Conv2D(16, (3, 3), activation='relu')(input_data)
    # x = layers.MaxPooling2D()(x)
    # x = layers.Conv2D(32, (3, 3), activation="relu")(x)
    # x = layers.MaxPooling2D(padding='same')(x)
    # x = layers.Flatten()(x)
    # x = layers.Dense(64, activation='relu')(x)
    # x = layers.Dropout(0.2)(x)
    # model_output = layers.Dense(3, activation='softmax')(x)
    # model = keras.Model(input_data, model_output)
    # 训练曲线较稳定
    # input_data = keras.Input(shape=(625, 8, 1))
    # x = layers.Conv2D(8, (7, 7), activation='relu', padding='same')(input_data)
    # x = layers.MaxPooling2D()(x)
    # x = layers.Conv2D(32, (3, 3), activation="relu", padding='same')(x)
    # x = layers.MaxPooling2D(padding='same')(x)
    # x = layers.Flatten()(x)
    # x = layers.Dense(64, activation='relu')(x)
    # x = layers.Dropout(0.2)(x)
    # model_output = layers.Dense(3, activation='softmax')(x)
    # model = keras.Model(input_data, model_output)
    # return model

def bp_model():
    input_data = keras.Input(shape=(cfg.row, cfg.column, 1))
    x = layers.Flatten()(input_data)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    model_output = layers.Dense(cfg.category, activation='softmax')(x)
    model = keras.Model(input_data, model_output)
    return model


def cnn_rnn_model():
    # initialize the model along with the input shape to be "channels last" ordering
    input_wav = keras.Input(shape=(cfg.row, cfg.column, 1))
    x = layers.Conv2D(32, (3, 3), activation='relu')(input_wav)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, (3, 3), activation="relu")(x)
    x = layers.Reshape(target_shape=(309, 64))(x)
    x = layers.LSTM(128, return_sequences=False)(x)
    x = layers.Dropout(0.5)(x)
    model_output = layers.Dense(cfg.category)(x)
    model = keras.Model(input_wav, model_output)
    # return the constructed network architecture
    return model

def rnn_model():
    # initialize the model along with the input shape to be "channels last" ordering
    input_wav = keras.Input(shape=(cfg.row, cfg.column))
    x = layers.LSTM(32, return_sequences=True)(input_wav)
    x = layers.LSTM(64, return_sequences=True)(x)
    x = layers.Dropout(0.5)(x)
    # x = layers.LSTM(64, return_sequences=True)(x)
    # x = layers.LSTM(64, return_sequences=True)(x)
    model_output = layers.LSTM(cfg.category, activation='relu', return_sequences=False)(x)
    model = keras.Model(input_wav, model_output)
    # return the constructed network architecture
    return model



if __name__ == '__main__':
    model = cnn_model()
    # 打印神经网络结构，统计参数数目
    model.summary()
