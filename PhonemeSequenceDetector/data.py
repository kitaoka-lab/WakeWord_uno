#!/usr/bin/env python3

# Copyright 2021 Toyohashi University of Technology (Yuki Sensui)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


import argparse
from pathlib import Path
import math
import numpy as np
import torch
import tensorflow as tf
import xml.etree.ElementTree as ET

"""
# CONFIG
"""
# Wake Wordの位置から、ラベルを1にするフレーム数
FRAME_LENGTH_WAKE_WORD_DETECT = 10
TOKEN_LIST = ['<blank>', '<unk>', 'a', 'o', 'i', 'e', 'u', 't', 'k', 'H', 'n', 'm', 'r', 's', 'N', 'd', 'g', 'sj', 'Q', 'y', 'nj', 'w', 'kj', 'b', 'h',
              'c', 'cj', 'zj', 'sy', 'zy', 'z', 'hj', 'p', 'cy', 'F', 'ky', 'gj', 'ry', 'hy', 'FV', 'gy', 'ty', 'ny', 'by', 'py', '?', 'my', 'dy', '<sos/eos>']
BLANK_IDX = 0

PROB_BASE_PATH = Path('/home/unotakaaki/espnet/egs2/csj/asr1/prob_dump_2')
XML_BASE_PATH = Path('/home/unotakaaki/CSJ/XML/BaseXML/core')
XML_EXT = '.xml'
DATASET_SAVE_BASE_PATH = Path.cwd() / 'data_frame_10'
# DATASET_SAVE_BASE_PATH = Path.cwd() / 'data'


def pick_phonemes_from_posterior(*, posterior, phonemes, frame_num, length):
    data = np.empty((frame_num, length + 1))
    # print(data)
    # print(data.shape) #(frame,ww_length + 1)
    # print(posterior)
    # print(posterior.shape) #(frame,49)
    for idx, prev in enumerate(posterior):
        # print(prev)
        # print(prev.shape) #(49,)
        frame = []
        for phoneme in phonemes: # phonemes:wwのリスト.例['a','b','c']
            assert phoneme != None # assert 条件式, 条件式がFalseの場合に出力するメッセージ
            token_idx = TOKEN_LIST.index(phoneme) # 音素のidxをtokenlistから取り出しtoken_idxへ
            frame = np.append(frame, prev[token_idx]) #wwの音素確率をリストに追加していく
        frame = np.append(frame, prev[BLANK_IDX]) # blankの確率をリストの末尾に追加
        data[idx] = frame
    return data


def prepare_data(length, set_type):
    x = []
    y = []
    fra=[]
    last_xml_path = ''
    xml_tree = None
    ipu_iter = None

    prob_path = PROB_BASE_PATH / set_type
    for path in sorted(prob_path.glob('*.pt')): # .glob 正規表現
        prob = torch.load(path).squeeze(0).to('cpu').detach().numpy().copy()
        frame_num = prob.shape[0]
        [id, start_str, end_str] = path.stem.split('_') #stemディレクトリ名取得　split _で分割
        start = float(start_str) / 1000
        end = float(end_str) / 1000
        utt_duration = end - start

        # If an ID starts with `sp`, it is speed pertubated data. Pick utt ID.
        # Ex: sp0.9-A03M0018
        if id.startswith('sp'):
            id = id.split('-')[1]

        data = []
        labels = np.zeros(shape=(frame_num))
        # print(labels.shape) #(frame,) [0,0,0,...,0,0]

        xml_path = XML_BASE_PATH / (id + XML_EXT)
        if xml_path != last_xml_path:
            last_xml_path = xml_path
            xml_tree = ET.parse(xml_path)
            root = xml_tree.getroot()
            ipu_iter = iter(root.findall('IPU')) #リストやタプルなど複数の要素を持つデータに対して、イテレータに変換することで要素を最初から順番に取り出すことができる
            # ipuをrootとするtreeのイテレータを作成

        while True:
            ipu = next(ipu_iter)
            ipu_start = float(ipu.get('IPUStartTime'))
            ipu_end = float(ipu.get('IPUEndTime'))
            if ipu_start < start:
                continue

            # Find the first word whose phoneme length is the target length
            is_found = False
            for word_el in ipu.findall('.//LUW'):
                phoneme_els = word_el.findall('.//Phoneme')
                if len(phoneme_els) != length: #指定した文字数の単語を探す
                    continue
                is_found = True

                phonemes = [el.get('PhonemeEntity') for el in phoneme_els] #wwのリスト作成.例['a','b','c']                # Get the word end time
                last_phoneme_el = phoneme_els[len(phoneme_els) - 1] # wwの最後の文字を代入
                phone_els = last_phoneme_el.findall('.//Phone')
                word_end = float(
                    phone_els[len(phone_els) - 1].get('PhoneEndTime')) - start

                # Pick the probs of phonemes in the target word and prob of blank from `probs` as `data`
                data = pick_phonemes_from_posterior(
                    posterior=prob, phonemes=phonemes, frame_num=frame_num, length=length)
                data = np.empty((frame_num, length + 1))
                # print(data.shape) #(frame,ww_length + 1)
                # print(posterior.shape) #(frame,49)
                for idx, prev in enumerate(prob):
                    # print(prev.shape) #(49,)
                    frame = []
                    for phoneme in phonemes: # phonemes:wwのリスト.例['a','b','c']
                        assert phoneme != None # assert 条件式, 条件式がFalseの場合に出力するメッセージ
                        token_idx = TOKEN_LIST.index(phoneme) # 音素のidxをtokenlistから取り出しtoken_idxへ
                        frame = np.append(frame, prev[token_idx]) #wwの音素確率をリストに追加していく
                    frame = np.append(frame, prev[BLANK_IDX]) # blankの確率をリストの末尾に追加
                    data[idx] = frame
                # print(data)
                # print(data.shape) #(frame,ww_length + 1)
                # Mark some frames after the end of word as "detected wake word"
                word_idx = math.ceil(word_end / utt_duration * frame_num)
                # print(word_idx)
                # labels[word_idx:] = 1
                labels[word_idx: word_idx + FRAME_LENGTH_WAKE_WORD_DETECT] = 1
                # print(labels.shape) (frame,)

                break

            if is_found or ipu_end >= end:
                break

        # If no word whose phoneme length is target length was found, set dummy data
        if len(data) == 0:
            phonemes = [TOKEN_LIST[np.random.randint(
                1, len(TOKEN_LIST))] for _ in range(length)] # ダミーのwwを作成:例['a','b','c']
            data = pick_phonemes_from_posterior(
                posterior=prob, phonemes=phonemes, frame_num=frame_num, length=length)

        x.append(data) # numpy配列をリストに追加.大きさはバラバラ
        y.append(labels) # numpy配列をリストに追加.大きさはバラバラ
    # print("x\n")   
    # print(x)
    # print("y\n")
    # print(y)
    # print(str(len(x))+","+str(len(x[0]))+","+str(len(x[0][0]))) #(ファイル数, フレーム,ww_length+1)
    # print(str(len(y))+","+str(len(y[0]))) #(ファイル数, フレーム)
    x = tf.ragged.constant(x)
    y = tf.ragged.constant(y)
    # print("x\n")   
    # print(x)
    # print("y\n")
    # print(y)
    return x, y


if __name__ == "__main__":
    parser = argparse.ArgumentParser() # パーサーを作る。パーサーとは構文解析を行うプログラムや処理のこと
    parser.add_argument("length", type=int, help="target phoneme length") #整数のリストを受け取って合計か最大値を返す
    parser.add_argument("settype", type=str,
                        help="all, train_dev, train_nodev, or eval")
    args = parser.parse_args()
    length = args.length
    set_type = args.settype

    if set_type == 'train_dev' or set_type == 'all':
        x, y = prepare_data(length, 'train_dev') #データセットの準備
        # print(x)
        # print(y)
        # print(x.shape) #(ファイル数,none,none)
        # print(y.shape) #(ファイル数,none)
        # x = x.to_tensor(1.)
        # y = y.to_tensor(1.)
        x = x.to_tensor(0.)
        y = y.to_tensor(0.)
        # print(x[0])
        # print(y[0])
        # print(x.shape) #(ファイル数,max frame数,ww音素数+1)
        # print(y.shape) #(ファイル数,max frame数)

        save_dir = DATASET_SAVE_BASE_PATH / str(length) / 'train_dev'
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        np.save(save_dir / 'x.npy', x)
        np.save(save_dir / 'y.npy', y)
        print('Saved to the files in: ' + str(save_dir))
    if set_type == 'train_nodev' or set_type == 'all':
        x, y = prepare_data(length, 'train_nodup')
        # x = x.to_tensor(1.)
        # y = y.to_tensor(1.)
        x = x.to_tensor(0.)
        y = y.to_tensor(0.)
        # print(x.shape) #(ファイル数,max frame数,ww音素数+1)
        # print(y.shape) #(ファイル数,max frame数)

        save_dir = DATASET_SAVE_BASE_PATH / str(length) / 'train_nodev'
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        np.save(save_dir / 'x.npy', x)
        np.save(save_dir / 'y.npy', y)
        print('Saved to the files in: ' + str(save_dir))
    if set_type == 'eval' or set_type == 'all':
        x_eval1, y_eval1 = prepare_data(length, 'eval1')
        x_eval2, y_eval2 = prepare_data(length, 'eval2')
        x_eval3, y_eval3 = prepare_data(length, 'eval3')
        # print(x_eval1.shape)
        # print(x_eval2.shape)
        # print(x_eval3.shape)
        # x = tf.concat([x_eval1, x_eval2, x_eval3], 0).to_tensor(1.)
        # y = tf.concat([y_eval1, y_eval2, y_eval3], 0).to_tensor(1.)
        x = tf.concat([x_eval1, x_eval2, x_eval3], 0).to_tensor(0.)
        y = tf.concat([y_eval1, y_eval2, y_eval3], 0).to_tensor(0.)
        # print(x.shape) #(ファイル数,max frame数,ww音素数+1)
        # print(y.shape) #(ファイル数,max frame数)
        # print(y[0])

        save_dir = DATASET_SAVE_BASE_PATH / str(length) / 'eval'
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        np.save(save_dir / 'x.npy', x)
        np.save(save_dir / 'y.npy', y)
        print('Saved to the files in: ' + str(save_dir))
