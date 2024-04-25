import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path

THRESHOLDS_NUM = 200
#THRESHOLD_LOWER_LIMIT = 0
THRESHOLD_LOWER_LIMIT = 0.45
THRESHOLD_UPPER_LIMIT = 1.


def false_rejection_rate(threshold=0.5):
    def calc(label, pred):
        # print("label\n")
        # print(label) # (32,623)
        # print("pred\n")
        # print(pred) # (32,623,2)
        label = tf.one_hot(tf.cast(label, tf.int64), 2)
        # print(label)
        # print(label.shape)

        pred = pred * [0, 1]
        pred = pred + [threshold, 0]
        pred = tf.one_hot(tf.math.argmax(
            pred, axis=2, output_type=tf.int64), 2)
        # print(pred)
        # print(pred.shape)

        label_positive = label * [0, 1]
        # print("label_positive")
        # print(label_positive) # (32, 581, 2)
        label_positive_sum = tf.math.reduce_sum(label_positive).numpy()
        # print("label_positive_sum")
        # print(label_positive_sum) 
        true_positive_sum = tf.math.reduce_sum(
            label_positive * pred).numpy()
        # print("true_positive_sum")
        # print(true_positive_sum) 
        return (label_positive_sum - true_positive_sum) / label_positive_sum
    calc.__name__ = 'frr'
    return calc


def false_alarm_rate(threshold=0.5):
    def calc(label, pred):
        label = tf.one_hot(tf.cast(label, tf.int64), 2)
        # tf.one_hot:教師ラベルに対応する要素だけが1で他は0であるOne hotベクトルを教師データとすることで学習させることができる。
        # depth=2:分類するクラスの数を指定する。
        # tf.cast( 変換したいもの , 変換後の型 )

        pred = pred * [0, 1]
        pred = pred + [threshold, 0]
        pred = tf.one_hot(tf.math.argmax(
            pred, axis=2, output_type=tf.int64), 2)

        label_negative = label * [1, 0]
        label_negative = label_negative[:, :, ::-1]
        pred_positive = pred * [0, 1]
        label_negative_sum = tf.math.reduce_sum(label_negative).numpy() + 1e-07
        # print("label_negative_sum")
        # print(label_negative_sum) 
        false_positive_sum = tf.math.reduce_sum(
            label_negative * pred_positive).numpy()
        # print("false_positive_sum")
        # print(false_positive_sum) 
        return false_positive_sum / label_negative_sum
    calc.__name__ = 'far'
    return calc


def generate_det_curve(label, pred):
    points = []
    for threshold in np.linspace(THRESHOLD_LOWER_LIMIT, THRESHOLD_UPPER_LIMIT, num=THRESHOLDS_NUM):
        frr = false_rejection_rate(threshold)(label, pred)
        far = false_alarm_rate(threshold)(label, pred)
        points.append([frr, far, threshold])
    points = sorted(points, key=lambda k: k[0]) #frrでsort
    points = np.array(points)
    # print(points)
    # print(points.shape)
    x = points[:, 0] #全てのFRRを入れる
    y = points[:, 1] #全てのFARを入れる

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x, y)
    ax.set_xlabel('FRR', fontsize=14)
    ax.set_ylabel('FAR', fontsize=14)
    ax.set_xmargin(0)
    ax.set_ymargin(0)
    #ax.set_xlim(left=(ax.get_xlim()[0] - 0.075)) # set_xlim: x 軸方向の表示範囲を指定する
    plt.tight_layout() # サブプロット間の正しい間隔を自動的に維持します。

    # plt.xlim(left=0.)

    save_dir = Path.cwd() / 'exp_20221209_frame_10'
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    plt.savefig('exp_20221209_frame_10/det_curve.png')
    np.save(save_dir / 'det_curve_points.npy', points)
