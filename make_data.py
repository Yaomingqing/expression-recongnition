import numpy as np
import os
import cv2
import random
import matplotlib.pyplot as plt
import h5py


WIDTH = 48
HEIGHT = 48
SCALE_BASE = 10
DATA_ADD = False
TRAIN_DIR = 'data/train/image_data/'
TEST_DIR = 'data/test/JAFFE/jatra/'

# 对图片进行仿射变换，就是平移、旋转、缩放
def affine_trans(image):
    MM = np.zeros((2, 3))
    tx = np.random.random_integers(-4, 4, 1)
    ty = np.random.random_integers(-4, 4, 1)
    degree = np.random.random_integers(-5, 5, 1)
    scale = np.random.random_integers(-2, 2, 1)
    scale = float(scale + SCALE_BASE) / float(SCALE_BASE)
    MM = cv2.getRotationMatrix2D((HEIGHT/2, WIDTH/2), degree, scale)
    MM[0, 2] += tx
    MM[1, 2] += ty
    InMM = cv2.invertAffineTransform(MM)

    return cv2.warpAffine(image, InMM, (HEIGHT, WIDTH))

# 对图片进行增加高斯噪声
def addgaussian(image, percentage):
    gaussianimg = image
    guassian_num = int(percentage*image.shape[0]*image.shape[1])
    for i in range(guassian_num):
        temp_x = np.random.randint(0, image.shape[0])
        temp_y = np.random.randint(0, image.shape[1])
        gaussianimg[temp_x][temp_y] = 255

    return gaussianimg

# 改变图片对比度、亮度
def contrast_and_light(image):
    scale = np.random.random_integers(-2, 2, 1)
    scale = float(scale + SCALE_BASE) / float(SCALE_BASE)
    newimg = image * scale

    return newimg

# 对图片进行遮挡一部分
def shield(image):
    nre_img = image
    tx = np.random.randint(2, 5, 1)
    ty = np.random.randint(2, 5, 1)
    x = np.random.randint(tx, image.shape[0]-tx, 1)
    y = np.random.randint(ty, image.shape[0]-ty, 1)
    tx, ty, x, y = int(tx), int(ty), int(x), int(y)
    # nre_img[x-tx:x+tx][y-ty:y+ty] = np.zeros((2*tx, 2*ty))
    for i in range(x-tx, x+tx):
        # print(i, type(i))
        for j in range(y-ty, y+ty):
            nre_img[i][j] = 0

    return nre_img


# make training data
def make_training_data(train_dir):
    training_data = []
    training_data_label = []
    for thedir in os.listdir(train_dir):
        print(thedir)
        if thedir == 'neutral-0':
            folder = os.path.join(TRAIN_DIR, thedir)
            for dirpath, dirnames, filenames in os.walk(folder):
                for file in filenames:
                    one_image_dir = os.path.join(dirpath, file)
                    image = cv2.imread(one_image_dir, 0)
                    image = cv2.resize(image, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
                    # add origin image
                    image1 = image / 256
                    training_data.append(image1)
                    # one-hot label
                    training_data_label.append([1, 0, 0, 0, 0, 0, 0])
                    if DATA_ADD:
                        # add affine transform image
                        image2 = affine_trans(image)/256
                        training_data.append(image2)
                        training_data_label.append([1, 0, 0, 0, 0, 0, 0])
                        # add gaussian noise
                        image3 = addgaussian(image, 0.01) / 256
                        training_data.append(image3)
                        training_data_label.append([1, 0, 0, 0, 0, 0, 0])

                        # change image light
                        image4 = contrast_and_light(image/256)
                        training_data.append(image4)
                        training_data_label.append([1, 0, 0, 0, 0, 0, 0])

                        # shield image
                        image5 = shield(image)/256
                        training_data.append(image5)
                        training_data_label.append([1, 0, 0, 0, 0, 0, 0])

                        # cv2.imshow('image', image)
                        # cv2.imshow('image1', image1)
                        # cv2.imshow('image2', image2)
                        # cv2.imshow('image3', image3)
                        # cv2.imshow('image4', image4)
                        # cv2.imshow('image5', image5)
                        # cv2.waitKey()
                        # cv2.imwrite('image/'+file+'image.png', image)
                        # cv2.imwrite('image/'+file+'image1.png', image1*255)
                        # cv2.imwrite('image/'+file+'image2.png', image2*255)
                        # cv2.imwrite('image/'+file+'image3.png', image3*255)
                        # cv2.imwrite('image/'+file+'image4.png', image4*125)
                        # cv2.imwrite('image/'+file+'image5.png', image5*255)


        elif thedir == 'happy-1':
            folder = os.path.join(TRAIN_DIR, thedir)
            for dirpath, dirnames, filenames in os.walk(folder):
                # print(dirpath)
                # print(dirnames)
                # print(filenames)
                for file in filenames:
                    one_image_dir = os.path.join(dirpath, file)
                    image = cv2.imread(one_image_dir, 0)
                    image = cv2.resize(image, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
                    label = np.array([0, 1, 0, 0, 0, 0, 0])
                    # add origin image
                    image1 = image / 256
                    training_data.append(image1)
                    # one-hot label
                    training_data_label.append(label)
                    if DATA_ADD:
                        # add affine transform image
                        image2 = affine_trans(image) / 256
                        training_data.append(image2)
                        training_data_label.append(label)
                        # add gaussian noise
                        image3 = addgaussian(image, 0.01) / 256
                        training_data.append(image3)
                        training_data_label.append(label)

                        # change image light
                        image4 = contrast_and_light(image / 256)
                        training_data.append(image4)
                        training_data_label.append(label)

                        # shield image
                        image5 = shield(image)/256
                        training_data.append(image5)
                        training_data_label.append(label)


        elif thedir == 'surprise-2':
            folder = os.path.join(TRAIN_DIR, thedir)
            for dirpath, dirnames, filenames in os.walk(folder):
                for file in filenames:
                    one_image_dir = os.path.join(dirpath, file)
                    image = cv2.imread(one_image_dir, 0)
                    image = cv2.resize(image, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
                    label = np.array([0, 0, 1, 0, 0, 0, 0])
                    # add origin image
                    image1 = image / 256
                    training_data.append(image1)
                    # one-hot label
                    training_data_label.append(label)
                    if DATA_ADD:
                        # add affine transform image
                        image2 = affine_trans(image) / 256
                        training_data.append(image2)
                        training_data_label.append(label)
                        # add gaussian noise
                        image3 = addgaussian(image, 0.01) / 256
                        training_data.append(image3)
                        training_data_label.append(label)

                        # change image light
                        image4 = contrast_and_light(image / 256)
                        training_data.append(image4)
                        training_data_label.append(label)

                        # shield image
                        image5 = shield(image)/256
                        training_data.append(image5)
                        training_data_label.append(label)
        elif thedir == 'angry-3':
            folder = os.path.join(TRAIN_DIR, thedir)
            for dirpath, dirnames, filenames in os.walk(folder):
                for file in filenames:
                    one_image_dir = os.path.join(dirpath, file)
                    image = cv2.imread(one_image_dir, 0)
                    image = cv2.resize(image, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
                    label = np.array([0, 0, 0, 1, 0, 0, 0])
                    # add origin image
                    image1 = image / 256
                    training_data.append(image1)
                    # one-hot label
                    training_data_label.append(label)
                    if DATA_ADD:
                        # add affine transform image
                        image2 = affine_trans(image) / 256
                        training_data.append(image2)
                        training_data_label.append(label)
                        # add gaussian noise
                        image3 = addgaussian(image, 0.01) / 256
                        training_data.append(image3)
                        training_data_label.append(label)

                        # change image light
                        image4 = contrast_and_light(image / 256)
                        training_data.append(image4)
                        training_data_label.append(label)

                        # shield image
                        image5 = shield(image)/256
                        training_data.append(image5)
                        training_data_label.append(label)

        elif thedir == 'disgust-4':
            folder = os.path.join(TRAIN_DIR, thedir)
            for dirpath, dirnames, filenames in os.walk(folder):
                for file in filenames:
                    one_image_dir = os.path.join(dirpath, file)
                    image = cv2.imread(one_image_dir, 0)
                    image = cv2.resize(image, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
                    label = np.array([0, 0, 0, 0, 1, 0, 0])
                    # add origin image
                    image1 = image / 256
                    training_data.append(image1)
                    # one-hot label
                    training_data_label.append(label)
                    if DATA_ADD:
                        # add affine transform image
                        image2 = affine_trans(image) / 256
                        training_data.append(image2)
                        training_data_label.append(label)
                        # add gaussian noise
                        image3 = addgaussian(image, 0.01) / 256
                        training_data.append(image3)
                        training_data_label.append(label)

                        # change image light
                        image4 = contrast_and_light(image / 256)
                        training_data.append(image4)
                        training_data_label.append(label)

                        # shield image
                        image5 = shield(image)/256
                        training_data.append(image5)
                        training_data_label.append(label)

        elif thedir == 'fear-5':
            folder = os.path.join(TRAIN_DIR, thedir)
            for dirpath, dirnames, filenames in os.walk(folder):
                for file in filenames:
                    one_image_dir = os.path.join(dirpath, file)
                    image = cv2.imread(one_image_dir, 0)
                    image = cv2.resize(image, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
                    label = np.array([0, 0, 0, 0, 0, 1, 0])
                    # add origin image
                    image1 = image / 256
                    training_data.append(image1)
                    # one-hot label
                    training_data_label.append(label)
                    if DATA_ADD:
                        # add affine transform image
                        image2 = affine_trans(image) / 256
                        training_data.append(image2)
                        training_data_label.append(label)
                        # add gaussian noise
                        image3 = addgaussian(image, 0.01) / 256
                        training_data.append(image3)
                        training_data_label.append(label)

                        # change image light
                        image4 = contrast_and_light(image / 256)
                        training_data.append(image4)
                        training_data_label.append(label)

                        # shield image
                        image5 = shield(image)/256
                        training_data.append(image5)
                        training_data_label.append(label)

        else:
            folder = os.path.join(TRAIN_DIR, thedir)
            for dirpath, dirnames, filenames in os.walk(folder):
                for file in filenames:
                    one_image_dir = os.path.join(dirpath, file)
                    image = cv2.imread(one_image_dir, 0)
                    image = cv2.resize(image, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
                    label = np.array([0, 0, 0, 0, 0, 0, 1])
                    # add origin image
                    image1 = image / 256
                    training_data.append(image1)
                    # one-hot label
                    training_data_label.append(label)
                    if DATA_ADD:
                        # add affine transform image
                        image2 = affine_trans(image) / 256
                        training_data.append(image2)
                        training_data_label.append(label)
                        # add gaussian noise
                        image3 = addgaussian(image, 0.01) / 256
                        training_data.append(image3)
                        training_data_label.append(label)

                        # change image light
                        image4 = contrast_and_light(image / 256)
                        training_data.append(image4)
                        training_data_label.append(label)

                        # shield image
                        image5 = shield(image)/256
                        # print(image5)
                        training_data.append(image5)
                        training_data_label.append(label)

    training_data = np.asarray(training_data)
    training_data_label = np.asarray(training_data_label)



    # shuffle the order
    index = list(range(len(training_data)))
    random.shuffle(index)
    training_data = training_data[index]
    training_data_label = training_data_label[index]

    print('training_data_label.shape:', training_data_label.shape, type(training_data_label))
    print('training_data.shape:', training_data.shape, type(training_data))

    return training_data, training_data_label

# make test data
def make_test_data(test_dir):
    test_data = []
    test_data_label = []

    for i in range(7):
        dir = os.path.join(test_dir, str(i))
        # print(dir)
        # print(os.listdir(dir))
        for file in os.listdir(dir):
            image_dir = os.path.join(dir, file)
            # print(image_dir)
            image = cv2.imread(image_dir, 0)
            image = cv2.resize(image, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
            label = np.zeros(7)
            label[i] = 1
            # add origin image
            image1 = image / 256
            test_data.append(image1)
            # one-hot label
            test_data_label.append(label)
            if DATA_ADD:
                # add affine transform image
                image2 = affine_trans(image) / 256
                test_data.append(image2)
                test_data_label.append(label)
                # add gaussian noise
                image3 = addgaussian(image, 0.01) / 256
                test_data.append(image3)
                test_data_label.append(label)

                # change image light
                image4 = contrast_and_light(image / 256)
                # print(image4)
                test_data.append(image4)
                test_data_label.append(label)

                # shield image
                image5 = shield(image)/256
                test_data.append(image5)
                test_data_label.append(label)


    test_data = np.asarray(test_data)
    test_data_label = np.asarray(test_data_label)
    # shuffle the order
    index = list(range(len(test_data)))
    random.shuffle(index)
    test_data = test_data[index]
    test_data_label = test_data_label[index]

    print('test_data_label.shape:', test_data_label.shape)
    print('test_data.shape:', test_data.shape)

    return test_data, test_data_label


def make_data(data, label, savepath):
    with h5py.File(savepath, 'w') as hf:
        hf.create_dataset('data', data=data)
        hf.create_dataset('label', data=label)


if __name__ == '__main__':
    training_data_dir = 'training.h5'
    test_data_dir = 'test.h5'
    if DATA_ADD:
        training_data_dir = 'training_add.h5'
        test_data_dir = 'test_add.h5'
    training_data, training_data_label = make_training_data(TRAIN_DIR)
    make_data(training_data, training_data_label, training_data_dir)
    test_data, test_data_label = make_test_data(TEST_DIR)
    make_data(test_data, test_data_label, test_data_dir)

