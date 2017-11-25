from mxnet.gluon import nn
from mxnet import init
from mxnet import gluon

import utils

class alexNet():
    def __init__(self):
        self.net = nn.Sequential()
        with self.net.name_scope():
            self.net.add(
                # First Level
                nn.Conv2D(channels=96, kernel_size=11,
                          strides=4, activation='relu'),
                nn.MaxPool2D(pool_size=3, strides=2),
                # Second Level
                nn.Conv2D(channels=256, kernel_size=5,
                          padding=2, activation='relu'),
                nn.MaxPool2D(pool_size=3, strides=2),
                # Third Level
                nn.Conv2D(channels=384, kernel_size=3,
                          padding=1, activation='relu'),
                nn.Conv2D(channels=384, kernel_size=3,
                          padding=1, activation='relu'),
                nn.Conv2D(channels=256, kernel_size=3,
                          padding=1, activation='relu'),
                nn.MaxPool2D(pool_size=3, strides=2),
                # Fourth Level
                # Convert input matrix to vector
                nn.Flatten(),
                nn.Dense(4096, activation="relu"),
                nn.Dropout(.5),
                # Fifth Level
                nn.Dense(4096, activation="relu"),
                nn.Dropout(.5),
                # Sixth Level
                nn.Dense(10)
            )


train_data, test_data = utils.load_data_fashion_mnist(batch_size=64, resize=224)
net = alexNet()
net.net.initialize(ctx=utils.try_gpu(), init=init.Xavier())
loss = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.net.collect_params(), 'sgd', {'learning_rate': 0.01})
utils.train(train_data, test_data, net.net, loss, trainer, utils.try_gpu(), num_epochs=5)
