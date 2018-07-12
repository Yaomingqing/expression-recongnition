import tensorflow as tf
import cv2
import numpy as np
import h5py
import os
# pip install -U sickit-learn
from sklearn.decomposition import PCA
from sklearn import svm

HEIGHT = 48
WIDTH = 48

DATA_ADD = True
feature_map_dir = 'result/'

# give a path, create the folder
def mkdir(path):
    folder = os.path.exists(path)

    if not folder:
        os.makedirs(path)

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


def inference(input):
    # our network
    # 卷积层
    conv1 = tf.layers.conv2d(inputs=input, filters=64, kernel_size=5, strides=1, padding='same',
                             activation=tf.nn.relu, name='conv1')
    # pooling层
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=3, strides=2, name='pooling1')
    conv2 = tf.layers.conv2d(pool1, 64, 3, 1, 'same', activation=tf.nn.relu, name='conv2')
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=3, strides=2, name='pooling2')
    conv3 = tf.layers.conv2d(pool2, 128, 3, 1, 'same', activation=tf.nn.relu, name='conv3')
    pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=3, strides=2, name='pooling3')
    print('parameters: ')
    print('pool3.shape: ', pool3.shape)
    # 全连接层
    fc1 = tf.layers.dense(tf.layers.flatten(pool3), 1024, activation=tf.nn.relu, name='fc1')
    fc2 = tf.layers.dense(fc1, 1024, activation=tf.nn.relu, name='fc2')
    # 输出层 feature layer, a image has 7 features
    output = tf.layers.dense(fc2, 7, activation=tf.nn.softmax, name='fc3')
    print('output.shape:', output.shape)

    return output, conv1, conv2, conv3

# 使用cnn模型提取图片特征
def feature_cnn(data):
    with tf.Session() as sess:
        # 定义输入输出变量形式
        x = tf.placeholder(tf.float32, [None, HEIGHT, WIDTH, 1], name='x-input')
        # 标签y_
        y_ = tf.placeholder(tf.float32, [None, 7], name='y-output')
        out, conv1, conv2, conv3 = inference(x)
        print('feature.shape, conv1.shape, conv2.shape, conv3.shape: ', out.shape, conv1.shape, conv2.shape, conv3.shape)
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint('./model/'))

        # var = tf.global_variables()
        # for idx, v in enumerate(var):
        #     print('param {:3}: {:5} {}'.format(idx, str(v.get_shape()), v.name))

        new_cnn_img_feature = []
        for i in range(len(data)):
            image = data[i]
            # print(image.shape)
            feed_dict = {x: np.resize(image, [1, WIDTH, HEIGHT, 1])}
            # 保存20个表情的特征图片，太多不好保存
            if i < 20:
                image_dir = feature_map_dir + str(i)
                mkdir(image_dir)
                # cv2.imshow('a', image)
                # cv2.waitKey()
                cv2.imwrite(image_dir+'/image_'+str(i)+'.png', image*255)

            # prob图片特征，c1, c2, c3 为卷积层输出
            prob, c1, c2, c3 = sess.run([out, conv1, conv2, conv3], feed_dict=feed_dict)
            if i < 20:
                # conv1 feature map
                for s1 in range(c1.shape[3]):
                    map1 = c1[0, :, :, s1]
                    # map1 = np.asarray(c1[0, :, :, s]).astype(np.uint8)
                    conv1_dir = image_dir+'/conv1/'
                    mkdir(conv1_dir)
                    # print(np.max(map1))
                    cv2.imwrite(conv1_dir+'filter_'+str(s1)+'.png', 1255*map1)  # 1255保证输出图片亮度，不会全黑
                    # cv2.imshow('11111', map1)
                    # cv2.waitKey()
                # conv2 feature map
                for s2 in range(c2.shape[3]):
                    map2 = c2[0, :, :, s2]
                    # print(np.max(map2))
                    conv2_dir = image_dir+'/conv2/'
                    mkdir(conv2_dir)
                    cv2.imwrite(conv2_dir+'filter_'+str(s2)+'.png', 1255*map2)
                # conv3 feature map
                for s3 in range(c3.shape[3]):
                    map3 = c3[0, :, :, s3]
                    # print(np.max(map3))
                    conv3_dir = image_dir+'/conv3/'
                    mkdir(conv3_dir)
                    cv2.imwrite(conv3_dir+'filter_'+str(s3)+'.png', 1255*map3)

            new_cnn_img_feature.append(prob)

        res = np.vstack(new_cnn_img_feature)
        print('cnn feature: ', res.shape)

        return res  # 返回所有图片cnn提取特征


def origin_feature(data):
    all_imgs = []
    for i in range(len(data)):
        image = data[i]
        # image = cv2.resize(image, (1, WIDTH*HEIGHT))
        image = image.flatten()
        # print(image.shape)
        all_imgs.append(image)
    res = np.array(all_imgs)
    # print(res.shape)

    return res  # 返回原图片，不提取任何特征，以原图作为输入


def pca_feature(data):
    print('origin images feature: ', data.shape)
    # 原图48*48=2304维特征
    # pca降维到128维
    pca = PCA(n_components=128)
    res = pca.fit_transform(data)
    print('pca feature: ', res.shape)
    return res

# 使用svm分类
def svm_classification(data, label, test_data, test_label):
    # define svm classification
    clf = svm.SVC()
    # train SVM
    clf.fit(data, label)
    # prediction
    # pre = clf.predict(test_data)
    # print(pre)

    # test accuracy
    score = clf.score(test_data, test_label)

    return score

def main():
    # 读取数据集
    training_data, training_data_label = read_data('training.h5')
    if DATA_ADD:
        training_data, training_data_label = read_data('training_add.h5')

    # svm不用太多数据，只取前2000样本
    training_data, training_data_label = training_data[:2000], training_data_label[:2000]
    # print(training_data.shape)
    # 标签变为实数
    labels = np.argmax(training_data_label, axis=1)

    # 三种情况：原图、cnn、pca，对比实验
    # 对原图处理
    origin_data = origin_feature(training_data)
    # 用cnn提取特征
    cnn_data = feature_cnn(training_data)
    # 用pca对原图进行降维，降低特征维度
    pca_data = pca_feature(origin_data)

    # 2000样本，八成训练，二成测试
    ratio = 0.8
    s = np.int(len(training_data) * ratio)

    # print(data.shape, labels.shape)

    # test_data, test_label = training_data[s:], training_data_label[s:]
    # print(test_data.shape, test_label.shape)
    origin_acc = svm_classification(origin_data[:s], labels[:s], origin_data[s:], labels[s:])
    cnn_acc = svm_classification(cnn_data[:s], labels[:s], cnn_data[s:], labels[s:])
    pca_acc = svm_classification(pca_data[:s], labels[:s], pca_data[s:], labels[s:])
    print('-------------------------------------------------')
    print('Use origin images, the accuracy is:  ', origin_acc)
    print('Use CNN feature, the accuracy is:    ', cnn_acc)
    print('Use PCA feature, the accuracy is:    ', pca_acc)


if __name__ == '__main__':
    main()

