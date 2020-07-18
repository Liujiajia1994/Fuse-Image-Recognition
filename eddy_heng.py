import os, sys
import pickle

import numpy as np
import tensorflow as tf
from keras import Input
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv1D, GlobalMaxPooling1D, Dense, LSTM
from keras.optimizers import Adam
from libtiff import TIFF
from keras.models import Model

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
# tensorflow按需求申请显存
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.Session(config=config)

shape_0 = 256
shape_1 = 256


def process_data():
    tokenizer_data = []
    for sets in read_all_data():
        item = {'image': [], 'label': []}
        for _set in sets:
            item['image'].append(_set[0])
            item['label'].append(_set[1])
        tokenizer_data.append(item)
    tokenizer_data = np.asarray(tokenizer_data)
    with open('./model/app.data', 'wb') as f:
        pickle.dump(tokenizer_data, f)
    return tokenizer_data


def read_all_data():
    all_tif = []
    read_data(all_tif, './eddy', [1, 0, 0])
    read_data(all_tif, './land', [0, 1, 0])
    read_data(all_tif, './sea_water', [0, 0, 1])
    all_tif = np.asarray(all_tif)
    np.random.seed(123)
    np.random.shuffle(all_tif)
    length = all_tif.shape[0]
    train = all_tif[:int(0.8 * length)]
    valid = all_tif[int(0.8 * length):int(0.9 * length)]
    test = all_tif[int(0.9 * length):]
    return train, valid, test


def read_data(all_tif, directory, label):
    file_list = os.listdir(directory)
    for file in file_list:
        image = TIFF.open(os.path.join(directory, file), mode="r").read_image()
        if shape_0 - image.shape[0] > 0:
            append_0 = np.zeros([shape_0 - image.shape[0], image.shape[1]])
            image = np.concatenate([image, append_0], axis=0)
        else:
            start = int((image.shape[0] - shape_0) / 2)
            image = image[start:start + shape_0]
        if shape_1 - image.shape[1] > 0:
            append_1 = np.zeros([shape_0, shape_1 - image.shape[1]])
            image = np.concatenate([image, append_1], axis=1)
        else:
            start = int((image.shape[1] - shape_1) / 2)
            image = image[:, start:start + shape_1]
        if image.shape[0] != shape_0 or image.shape[1] != shape_1:
            print('error')
        all_tif.append([np.divide(image, image.max()), label])
    return all_tif


def getCNN(file=None):
    inputs = Input(shape=(shape_0, shape_1), dtype='float32')
    model_layer1 = Conv1D(256, 2, padding='valid', activation='relu', strides=1)
    x_1 = model_layer1(inputs)
    x_1 = GlobalMaxPooling1D()(x_1)
    merged = Dense(256, activation='relu')(x_1)
    merged = Dense(64)(merged)
    merged = Dense(16)(merged)
    predictions = Dense(3, activation='softmax')(merged)
    model = Model(inputs=inputs,
                  outputs=predictions)
    if file is not None:
        model.load_weights(file)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    model.summary()
    return model


def run():
    try:
        with open('./model/app.data', 'rb') as f:
            tokenizer_data = pickle.load(f)
    except FileNotFoundError:
        print('processing data')
        tokenizer_data = process_data()
    train_set = tokenizer_data[0]
    valid_set = tokenizer_data[1]
    test_set = tokenizer_data[2]
    model = getCNN()
    checkpoint = ModelCheckpoint('./model/best.model', monitor='val_acc', save_best_only=True, mode='max', verbose=1,
                                 save_weights_only=True)
    hist = model.fit(np.asarray(train_set['image']), [train_set['label']], callbacks=[checkpoint],
                     validation_data=[np.asarray(valid_set['image']), [valid_set['label']]],
                     epochs=20, batch_size=64)
    model = getCNN('./model/best.model')
    predicts = model.predict(np.asarray(test_set['image']), batch_size=256)
    score, acc = model.evaluate(x=np.asarray(test_set['image']), y=[test_set['label']], batch_size=256)
    print("test acc:%s\t test score:%s\t history acc:%s\t history score:%s" % (
        acc, score, max(hist.history['acc']), min(hist.history['loss'])))


if __name__ == '__main__':
    run()
