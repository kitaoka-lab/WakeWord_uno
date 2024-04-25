#!/usr/bin/env python3

from io import open
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path
from tensorflow.python.ops.numpy_ops import np_config
from matplotlib import pyplot as plt
import os
import pandas as pd

from metrics import false_alarm_rate, false_rejection_rate, generate_det_curve

np_config.enable_numpy_behavior()

"""
## Hyperparameters
"""

learning_rate = 0.001
num_epochs = 100
dropout_rate = 0.0
lstm_layers = 1
lstm_units = 128
fc_units = 64


"""
## Prepare data
"""

num_phonemes = 6
num_classes = 2
input_shape = (None, num_phonemes + 1,)
batch_size = 4


DATASET_SAVE_BASE_PATH = Path.cwd() / 'data_frame_10'
# DATASET_SAVE_BASE_PATH = Path.cwd() / 'data'


def load_data(set_type):
    x = np.load(DATASET_SAVE_BASE_PATH /
                str(num_phonemes) / set_type / 'x.npy')
    y = np.load(DATASET_SAVE_BASE_PATH /
                str(num_phonemes) / set_type / 'y.npy')
    # return tf.data.Dataset.from_tensor_slices((x, y))
    return x, y


print("INFO: Preparing data")

x_train, y_train = load_data("train_dev")
print(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")

x_valid, y_valid = load_data("train_nodev")
print(f"x_valid shape: {x_valid.shape}, y_valid shape: {y_valid.shape}")

x_eval, y_eval = load_data("eval")
# print(f"x_eval shape: {x_eval.shape}, y_eval shape: {y_eval.shape}")


def create_model():
    inputs = layers.Input(shape=input_shape)
    h = layers.Masking(mask_value=0.)(inputs)

    # Create multiple layers of the LSTM block.
    for i in range(lstm_layers):
        h = layers.LSTM(lstm_units, return_sequences=True)(h) #lstm_units:出力の次元数 return_sequences:出力系列の最後の出力を返すか，完全な系列を返すか

    h = layers.Dense(fc_units, name="fc1", activation="relu")(h)
    logits = layers.Dense(num_classes, activation="softmax", name="outputs")(h)
    model = keras.Model(inputs=inputs, outputs=logits)
    return model

# from keras import backend as K
# def weighted_categorical_crossentropy(weights):
#     """
#     A weighted version of keras.objectives.categorical_crossentropy
    
#     Variables:
#         weights: numpy array of shape (C,) where C is the number of classes
    
#     Usage:
#         weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
#         loss = weighted_categorical_crossentropy(weights)
#         model.compile(loss=loss,optimizer='adam')
#     """
    
#     weights = K.variable(weights)
        
#     def loss(y_true, y_pred):
#         # scale predictions so that the class probas of each sample sum to 1
#         y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
#         # clip to prevent NaN's and Inf's
#         y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
#         # calc
#         loss = y_true * K.log(y_pred) * weights
#         loss = -K.sum(loss, -1)
#         return loss
    
#     return loss

def run_experiment(model):
    optimizer = keras.optimizers.Adam(learning_rate)
    # weights = np.array([0.2,0.8]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
    # loss = weighted_categorical_crossentropy(weights)

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy(
            name="accuracy"), false_rejection_rate(), false_alarm_rate()],
        run_eagerly=True,
    )
    # model.compile(
        # optimizer=optimizer,
        # loss=keras.losses.SparseCategoricalCrossentropy(),
        # metrics=[keras.metrics.SparseCategoricalAccuracy(
            # name="accuracy"), false_rejection_rate(), false_alarm_rate()],
        # run_eagerly=True,
    # )

    checkpoint_filepath = "tmp/checkpoint"
    # keras.callbacks.ModelCheckpoint:各エポック終了後にモデルを保存します．filepathは，（on_epoch_endで渡された）epochの値とlogsのキーで埋められた書式設定オプションを含むことができます.
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,
    ) # filepath: 文字列，モデルファイルを保存するパス．monitor: 監視する値.save_best_only: save_best_only=Trueの場合，監視しているデータによって最新の最良モデルが上書きされません．save_weights_only: Trueなら，モデルの重みが保存されます (model.save_weights(filepath))，そうでないなら，モデルの全体が保存されます (model.save(filepath))

    # Create a learning rate scheduler callback.
    # keras.callbacks.ReduceLROnPlateau:評価値を監視し，'patience'で指定されたエポック数の間改善が見られなかった場合，学習率を減らします．
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.2, patience=3
    ) # monitor: 監視する値．factor: 学習率を減らす割合．new_lr = lr * factor。patience: 何エポック改善が見られなかったら学習率の削減を行うか．

    # Create an early stopping callback.
    # tf.keras.callbacks.EarlyStopping:監視する値の変化が停止した時に訓練を終了します
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=15, restore_best_weights=True
    ) # monitor: 監視する値．patience: ここで指定したエポック数の間（監視する値に）改善がないと，訓練が停止します．
    
    # model.fit:固定回数（データセットの反復）の試行でモデルを学習させます
    history = model.fit(
        x=x_train,
        y=y_train,
        # batch_size=batch_size,
        epochs=num_epochs,
        validation_data=(x_valid, y_valid),
        # validation_split=0.1,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    ) # verbose: 整数．0，1，2のいずれか．進行状況の表示モード．0 = 表示なし，1 = プログレスバー，2 = 各試行毎に一行の出力．

    # model.load_weights(checkpoint_filepath)
    # model.evaluate:テストモードにおいて，モデルの損失値と評価値を返します．その計算はバッチ処理で行われます．
    _, accuracy, frr, far = model.evaluate(x_eval, y_eval)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")
    # print("batch_size:", batch_size)
    # print("dropout_rate:", dropout_rate)

    # model.predict:入力サンプルに対する予測の出力を生成します．その計算はバッチ処理で行われます．
    p_eval = model.predict(x_eval)
    # print(x_eval)
    print("y_eval\n")
    print(y_eval)
    print("p_eval\n")
    print(p_eval)
    print(x_eval.shape)
    print(y_eval.shape)
    print(p_eval.shape)
    generate_det_curve(y_eval, p_eval)
    np.save(Path.cwd() / 'data_frame_10' / str(num_phonemes) / 'eval/p.npy', p_eval)

    return history


print("INFO: Creating model")
model = create_model()
model.summary()
print("INFO: Running experiment")
history = run_experiment(model)
# # 生データの保存
# # history = model.fit(...)
# hist_df = pd.DataFrame(history.history)
# hist_df.to_csv('history.csv')

# # 可視化
# plt.figure()
# hist_df[['accuracy', 'val_accuracy']].plot()
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.savefig(os.path.join(self.result_dir, 'acc.png'))
# plt.close()

# plt.figure()
# hist_df[['loss', 'val_loss']].plot()
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.savefig(os.path.join(self.result_dir, 'loss.png'))
# plt.close()