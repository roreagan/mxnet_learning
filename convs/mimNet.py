from mxnet.gluon import nn
from mxnet import nd
from mxnet import gluon
from mxnet import init
import utils


def mlpconv(channels, kernel_size, padding,
            strides=1, max_pooling=True):
    out = nn.Sequential()
    out.add(
        nn.Conv2D(channels=channels, kernel_size=kernel_size,
                  strides=strides, padding=padding,
                  activation='relu'),
        nn.Conv2D(channels=channels, kernel_size=1,
                  padding=0, strides=1, activation='relu'),
        nn.Conv2D(channels=channels, kernel_size=1,
                  padding=0, strides=1, activation='relu'))
    if max_pooling:
        out.add(nn.MaxPool2D(pool_size=3, strides=2))
    return out

net = nn.Sequential()
# add name_scope on the outer most Sequential
with net.name_scope():
    net.add(
        mlpconv(96, 11, 0, strides=4),
        mlpconv(256, 5, 2),
        mlpconv(384, 3, 1),
        nn.Dropout(.5),
        # 10 target classes
        mlpconv(10, 3, 1, max_pooling=False),
        # Input: batch_size x 10 x 5 x 5 by AvgPooling
        # batch_size x 10 x 1 x 1
        nn.AvgPool2D(pool_size=5),
        # turn to batch_size x 10
        nn.Flatten()
    )


# blk = mlpconv(96, 11, 0, strides=4)
# blk.initialize()
#
# x = nd.random.uniform(shape=(32, 3, 224, 224))
# y = blk(x)
# print y.shape

train_data, test_data = utils.load_data_fashion_mnist(batch_size=64, resize=224)

ctx = utils.try_gpu()
net.initialize(ctx=ctx, init=init.Xavier())

loss = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.05})
utils.train(train_data, test_data, net, loss, trainer, ctx, num_epochs=10)
