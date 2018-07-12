import numpy as np
import h5py
import cv2
import matplotlib.pyplot as plt
from sklearn import datasets

DATA_ADD = True

def read_data(path):
    """
    Read h5 format data file
    Args:
        path: file path of desired file.
        data: '.h5' file format that contains train data values.
        label: '.h5' file format that contains train label values.
    """
    with h5py.File(path, 'r') as hf:
        data = np.array(hf.get('data'))
        label = np.array(hf.get('label'))
        return data, label


def main():
    training_data, training_data_label = read_data('training.h5')
    test_data, test_date_label = read_data('test.h5')
    if DATA_ADD:
        training_data, training_data_label = read_data('training_add.h5')
    test_data, test_date_label = read_data('test_add.h5')
    # training_data = training_data[:, :, :, np.newaxis]
    # print(training_data.shape, training_data_label.shape)
    # test_data = test_data[:, :, :, np.newaxis]

    ID=1
    print(training_data.shape)
    a = training_data[ID]
    print(a)
    # plt.imshow(a, cmap='gray')
    # plt.show()
    print(a.shape)
    # cv2.imshow('a', a)
    # cv2.waitKey()
    # print(a)
    a_label = training_data_label[ID]
    print(a_label)


if __name__ == '__main__':
    main()