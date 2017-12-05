from mxnet import image
from mxnet import nd
from mxnet import gluon


import utils

train_augs = [
    # Randomly Horizontal Flip
    image.HorizontalFlipAug(.5),
    # Randomly Crop
    image.RandomCropAug((28, 28))
]

test_augs = [
    image.CenterCropAug((28, 28))
]


# use multiple methods to expand and image
def apply_aug_list(img, augs):
    for f in augs:
        img = f(img)
    return img


def apply(img, aug, n=3):
    # picture should be changed to float
    # copy img n*n times
    X = [aug(img.astype('float32')) for _ in range(n*n)]

    Y = nd.stack(*X).clip(0, 255) / 255
    utils.show_images(Y, n, n, figsize=(8, 8))


def get_transform(augs):
    def transform(data, label):
        data = data.astype('float32')
        if augs is not None:
            data = nd.stack(*[apply_aug_list(d, augs) for d in data])
        data = nd.transpose(data, (0, 3, 1, 2))
        return data, label.astype('float32')
    return transform


def transform(data, label, augs):
    data = data.astype('float32')
    if augs is not None:
        for aug in augs:
            data = aug(data)
        nd.transpose(data, (2, 0, 1))


def get_data(batch_size, train_augs, test_augs=None):
    cifar10_train = gluon.data.vision.CIFAR10(train=True, transform=get_transform(train_augs))
    cifar10_test = gluon.data.vision.CIFAR10(train=False, transform=get_transform(test_augs))
    train_data = utils.DataLoader(cifar10_train, batch_size, shuffle=True)
    test_data = utils.DataLoader(cifar10_test, batch_size, shuffle=True)
    return (train_data, test_data)


def load_hotdog_pics(data_dir='E://deeplearningDatas/hotdog/'):
    train_imgs = gluon.data.vision.ImageFolderDataset(data_dir + 'hotdog/train',
                                                      transform=lambda X, y: get_transform(train_augs))
    test_imgs = gluon.data.vision.ImageFolderDataset(data_dir + 'hotdog/test',
                                                      transform=lambda X, y: get_transform(test_augs))
    train_data = gluon.data.DataLoader(train_imgs, 32, shuffle=True)
    test_data = gluon.data.DataLoader(test_imgs, 32)
    return train_data, test_data


