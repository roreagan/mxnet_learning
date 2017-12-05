from mxnet import gluon
from mxnet import init
from mxnet.gluon.model_zoo import vision as models

import zipfile
import picUtils
import utils

def get_hotdog_datas(data_dir='E://deeplearningDatas/hotdog'):
    fname = gluon.utils.download('https://apache-mxnet.s3-accelerate.amazonaws.com/gluon/dataset/hotdog.zip',
                                 path=data_dir, sha1_hash='fba480ffa8aa7e0febbb511d181409f899b9baa5')
    with zipfile.ZipFile(fname, 'r') as f:
        f.extract(data_dir)


def train(net, ctx, batch_size=64, epochs=100, learning_rate=0.01, wd=0.001, save_dir='./models/'):
    train_data, test_data = picUtils.load_hotdog_pics()
    net.collect_params().reset_ctx(ctx)
    net.hybridize()
    loss = gluon.loss.SoftmaxCrossEntropyLoss()

    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': learning_rate, "wd": wd})
    utils.train(train_data, test_data, net, loss, trainer, ctx, epochs)
    net.save_params(save_dir+'hotdog.params')


# get_hotdog_datas()

pretrained_net = models.resnet18_v2(pretrained=True, root='./premodels')

fitnet = models.resnet18_v2(classes=2)
fitnet.features = pretrained_net.features
fitnet.output.initialize(init.Xavier())

ctx = utils.try_gpu()
train(fitnet, ctx)
