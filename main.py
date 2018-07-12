import tensorflow as tf
import h5py
import numpy as np
import os
import matplotlib.pyplot as plt


os.environ['CUDA_VISIBLE_DEVICES'] = '7'
# dataset paramaters
# 图片大小
WIDTH = 48
HEIGHT = 48
# 控制数据增强
DATA_ADD = True
# control train or test
IS_TEST = True
# 类别
CLASSES = 7

# network parameters
# 训练步数
TRAINING_STEPS = 10000
# 每次训练使用的图片数量为100，组成一个batch
BATCH_SIZE = 100
# 学习率
LEARNING_RATE = 0.0001

# 神经网络
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
    print('pool3.shape: ', pool3.shape)
    # 全连接层
    fc1 = tf.layers.dense(tf.layers.flatten(pool3), 1024, activation=tf.nn.relu, name='fc1')
    fc2 = tf.layers.dense(fc1, 1024, activation=tf.nn.relu, name='fc2')
    # 输出层
    output = tf.layers.dense(fc2, CLASSES, activation=tf.nn.softmax, name='fc3')
    print('output.shape:', output.shape)

    return output, fc2  #输出为output类别预测概率， fc2为特征，用于center loss计算


def get_center_loss(features, labels, alpha, num_classes):
    """获取center loss及center的更新op

    Arguments:
        features: Tensor,表征样本特征,一般使用某个fc层的输出,shape应该为[batch_size, feature_length].
        labels: Tensor,表征样本label,非one-hot编码,shape应为[batch_size].
        alpha: 0-1之间的数字,控制样本类别中心的学习率,细节参考原文.
        num_classes: 整数,表明总共有多少个类别,网络分类输出有多少个神经元这里就取多少.

    Return：
        loss: Tensor,可与softmax loss相加作为总的loss进行优化.
        centers: Tensor,存储样本中心值的Tensor，仅查看样本中心存储的具体数值时有用.
        centers_update_op: op,用于更新样本中心的op，在训练时需要同时运行该op，否则样本中心不会更新
    """
    # 获取特征的维数，例如256维
    len_features = features.get_shape()[1]
    # print(len_features)
    # 建立一个Variable,shape为[num_classes, len_features]，用于存储整个网络的样本中心，
    # 设置trainable=False是因为样本中心不是由梯度进行更新的
    centers = tf.get_variable('centers', [num_classes, len_features], dtype=tf.float32,
                              initializer=tf.constant_initializer(0), trainable=False)
    # 将label展开为一维的，输入如果已经是一维的，则该动作其实无必要
    labels = tf.reshape(labels, [-1])

    # 根据样本label,获取mini-batch中每一个样本对应的中心值
    centers_batch = tf.gather(centers, labels)
    # 计算loss
    loss = tf.nn.l2_loss(features - centers_batch)

    # 当前mini-batch的特征值与它们对应的中心值之间的差
    diff = centers_batch - features

    # 获取mini-batch中同一类别样本出现的次数,了解原理请参考原文公式(4)
    unique_label, unique_idx, unique_count = tf.unique_with_counts(labels)
    appear_times = tf.gather(unique_count, unique_idx)
    appear_times = tf.reshape(appear_times, [-1, 1])

    diff = diff / tf.cast((1 + appear_times), tf.float32)
    diff = alpha * diff

    centers_update_op = tf.scatter_sub(centers, labels, diff)

    return loss, centers, centers_update_op

# This function takes an array of numbers and smoothes them out.
# Smoothing is useful for making plots a little easier to read.
def sliding_mean(data_array, window=10):
    new_list = []
    for i in range(len(data_array)):
        indices = range(max(i - window + 1, 0),
                        min(i + window + 1, len(data_array)))
        avg = 0
        for j in indices:
            avg += data_array[j]
        avg /= float(len(indices))
        new_list.append(avg)

    return new_list

# 训练网络和测试
def train_test(training_data,  training_data_label, test_data, test_date_label):
    # 定义输入输出变量形式
    x = tf.placeholder(tf.float32, [None, HEIGHT, WIDTH, 1], name='x-input')
    # 标签y_
    y_ = tf.placeholder(tf.float32, [None, CLASSES], name='y-output')
    global_step = tf.Variable(0, trainable=False, name='global_step')
    # print('------------', tf.argmax(y_, axis=1).shape)

    # 输入x，输出y
    y, feature = inference(x)

    # loss function
    # print(y.shape, y_.shape)
    # 普通分类loss
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_)
    loss = tf.reduce_mean(cross_entropy)
    # center loss
    center_loss, centers, centers_update_op = get_center_loss(feature, tf.argmax(y_, 1), 0.5, 7)
    # total loss
    total_loss = loss + 0.003*center_loss

    # 定义更新center loss参数和网络参数优化器
    with tf.control_dependencies([centers_update_op]):
        train_step = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(total_loss, global_step=global_step)
    # 计算预测准确率
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # 将所有数据分为训练集和验证集，训练集0.8，验证集0.2
    ratio = 0.8
    s = np.int(len(training_data) * ratio)
    train_date = training_data[:s]
    # print(train_date.shape)
    train_label = training_data_label[:s]

    validate_data = training_data[s:]
    validate_label = training_data_label[s:]

    # 开始循环训练， session是tensorflow的会话
    with tf.Session() as sess:
        # test
        # 定义测试集输入
        test_feed = {x: test_data, y_: test_date_label}
        if IS_TEST:
            saver = tf.train.Saver()
            print('Loading model............')
            saver.restore(sess, tf.train.latest_checkpoint('./model/origin_data_model'))  # model path
            test_acc = sess.run(accuracy, feed_dict=test_feed)
            print('\n--------Test accuracy using model is %g' % test_acc)
        else:
            # training and test
            # 初始化网络参数
            sess.run(tf.global_variables_initializer())
            # 定义验证集输入
            validate_feed = {
                x: validate_data,
                y_: validate_label
            }
            # 定义保存网络参数
            saver = tf.train.Saver()
            acc = []
            # 训练TRAINING_STEPS个batch
            for i in range(TRAINING_STEPS):

                if i % 10 == 0:
                    # 计算验证集准确率
                    validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                    acc.append(validate_acc)
                    print('after %d training steps, validation accuracy using model is %g' % (i, validate_acc))
                    # test_acc = sess.run(accuracy, feed_dict=test_feed)
                    # print('after %d training steps, test accuracy using average model is %g' % (i, test_acc))
                # 需要遍历所有训练集图片，每次取不同的batch
                start = (i*BATCH_SIZE) % len(train_date)
                end = min(start + BATCH_SIZE, len(train_date))
                # print('-------------', start, end)
                # 最重要的训练，这一步训练网络，更新网络参数
                sess.run(train_step, feed_dict={x: train_date[start:end], y_: train_label[start:end]})
            # 保存模型
            if DATA_ADD:
                save_path = 'model/origin_data_model'
            else:
                save_path = 'model/data_add_model'
            mkdir(save_path)
            saver.save(sess, save_path=save_path + '/model')

            # 在训练结束后，在测试数据集上检测神经网络模型的最终准确率
            test_acc = sess.run(accuracy, feed_dict=test_feed)
            print('\n--------After %d training steps, test accuracy using model is %g' % (TRAINING_STEPS, test_acc))
            # 平滑准确率，画图
            plot_acc = sliding_mean(acc)
            plt.figure()
            plt.plot(np.arange(len(plot_acc)), plot_acc)
            plt.xlabel('step')
            plt.ylabel('accuracy')
            plt.show()
# 创建目录
def mkdir(path):
    folder = os.path.exists(path)

    if not folder:
        os.makedirs(path)
    else:
        pass

# 读取训练数据集
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

# 主函数，读取数据集调用train_test训练网络
def main():
    training_data, training_data_label = read_data('training.h5')
    test_data, test_date_label = read_data('test.h5')
    if DATA_ADD:
        training_data, training_data_label = read_data('training_add.h5')
        test_data, test_date_label = read_data('test_add.h5')

    # 将所有数据分为训练集和测试集
    ratio = 0.9
    s = np.int(len(training_data) * ratio)
    train_data = training_data[:s]
    train_data_label = training_data_label[:s]
    test_data = training_data[s:]
    test_date_label = training_data_label[s:]
    train_data = train_data[:, :, :, np.newaxis]
    test_data = test_data[:, :, :, np.newaxis]

    print('Parameters:')
    print('-----train_data.shape: ', train_data.shape)
    print('-----train_data_label.shape', train_data_label.shape)
    print('-----test_data.shape: ', test_data.shape)
    print('-----test_date_label.shape', test_date_label.shape)

    train_test(train_data, train_data_label, test_data, test_date_label)


if __name__ == '__main__':
    main()