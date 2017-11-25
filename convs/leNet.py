from mxnet import nd
from mxnet import gluon
from mxnet import autograd as autograd

import utils



class simple_leNet():
    def __init__(self):
        weight_scale = .01
        ctx = utils.try_gpu()

        # output channels = 20, kernel = (5,5)
        self.W1 = nd.random_normal(shape=(20, 1, 5, 5), scale=weight_scale, ctx=ctx)
        self.b1 = nd.zeros(self.W1.shape[0], ctx=ctx)

        # output channels = 50, kernel = (3,3)
        self.W2 = nd.random_normal(shape=(50, 20, 3, 3), scale=weight_scale, ctx=ctx)
        self.b2 = nd.zeros(self.W2.shape[0], ctx=ctx)

        # output dim = 128
        self.W3 = nd.random_normal(shape=(1250, 128), scale=weight_scale, ctx=ctx)
        self.b3 = nd.zeros(self.W3.shape[1], ctx=ctx)

        # output dim = 10
        self.W4 = nd.random_normal(shape=(self.W3.shape[1], 10), scale=weight_scale, ctx=ctx)
        self.b4 = nd.zeros(self.W4.shape[1], ctx=ctx)

        self.params = [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3, self.W4, self.b4]
        for param in self.params:
            param.attach_grad()

    def net(self, X, verbose=False):
        X = X.as_in_context(self.W1.context)
        # First Conv Layer
        h1_conv = nd.Convolution(
            data=X, weight=self.W1, bias=self.b1, kernel=self.W1.shape[2:], num_filter=self.W1.shape[0])
        h1_activation = nd.relu(h1_conv)
        h1 = nd.Pooling(
            data=h1_activation, pool_type="avg", kernel=(2, 2), stride=(2, 2))
        # Second Conv Layer
        h2_conv = nd.Convolution(
            data=h1, weight=self.W2, bias=self.b2, kernel=self.W2.shape[2:], num_filter=self.W2.shape[0])
        h2_activation = nd.relu(h2_conv)
        h2 = nd.Pooling(data=h2_activation, pool_type="avg", kernel=(2, 2), stride=(2, 2))
        h2 = nd.flatten(h2)
        # First Full Connected
        h3_linear = nd.dot(h2, self.W3) + self.b3
        h3 = nd.relu(h3_linear)
        # Second Full Connected
        h4_linear = nd.dot(h3, self.W4) + self.b4
        if verbose:
            print('1st conv block:', h1.shape)
            print('2nd conv block:', h2.shape)
            print('1st dense:', h3.shape)
            print('2nd dense:', h4_linear.shape)
            print('output:', h4_linear)
        return h4_linear


batch_size = 256
train_data, test_data = utils.load_data_fashion_mnist(batch_size)

net = simple_leNet()

softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
learning_rate = .2

for epoch in range(5):
    train_loss = 0.
    train_acc = 0.
    for data, label in train_data:
        label = label.as_in_context(utils.try_gpu())
        with autograd.record():
            output = net.net(data)
            loss = softmax_cross_entropy(output, label)
        loss.backward()
        utils.SGD(net.params, learning_rate/batch_size)

        train_loss += nd.mean(loss).asscalar()
        train_acc += utils.accuracy(output, label)

    test_acc = utils.evaluate_accuracy(test_data, net.net, utils.try_gpu())
    print("Epoch %d. Loss: %f, Train acc %f, Test acc %f" % (
        epoch, train_loss/len(train_data),
        train_acc/len(train_data), test_acc))