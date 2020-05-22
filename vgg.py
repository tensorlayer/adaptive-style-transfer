#! /usr/bin/python
# -*- coding: utf-8 -*-
"""
VGG for ImageNet.

Introduction
----------------
VGG is a convolutional neural network model proposed by K. Simonyan and A. Zisserman
from the University of Oxford in the paper "Very Deep Convolutional Networks for
Large-Scale Image Recognition"  . The model achieves 92.7% top-5 test accuracy in ImageNet,
which is a dataset of over 14 million images belonging to 1000 classes.

Download Pre-trained Model
----------------------------
- Model weights in this example - vgg16_weights.npz : http://www.cs.toronto.edu/~frossard/post/vgg16/
- Model weights in this example - vgg19.npy : https://media.githubusercontent.com/media/tensorlayer/pretrained-models/master/models/
- Caffe VGG 16 model : https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md
- Tool to convert the Caffe models to TensorFlow's : https://github.com/ethereon/caffe-tensorflow

Note
------
- For simplified CNN layer see "Convolutional layer (Simplified)"
in read the docs website.
- When feeding other images to the model be sure to properly resize or crop them
beforehand. Distorted images might end up being misclassified. One way of safely
feeding images of multiple sizes is by doing center cropping.

"""

import os

import numpy as np
import tensorflow as tf

import tensorlayer as tl
from tensorlayer import logging
from tensorlayer.files import assign_weights, maybe_download_and_extract
from tensorlayer.layers import (BatchNorm, Conv2d, Dense, Flatten, Input, Lambda, LayerList, MaxPool2d,
                                DeConv2d, UpSampling2d, PadLayer, BatchNorm2d)
from tensorlayer.models import Model


_use_LayerList_before_merge = True
# Note: _use_DeConv2d will 1.bring Exception..(c.solved. b.happened again=PadLayer)
# AttributeError: The registered layer `layerlist_before_merge_3` should be built in advance. 
# Do you forget to pass the keyword argument 'in_channels'? -- tensorlayer\models\core.py", line 623, in __setattr__
_use_DeConv2d = True
_use_PadLayer_Reflect = True
_relu_class = tf.nn.relu  # tf.nn.relu, tf.nn.leaky_relu

__all__ = [
    'VGG',
    'vgg16',
    'vgg19',
    'vgg19_rev',
    'VGG16',
    'VGG19',
    #    'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    #    'vgg19_bn', 'vgg19',
]

layer_names = [
    ['conv1_1', 'conv1_2'], 'pool1', ['conv2_1', 'conv2_2'], 'pool2',
    ['conv3_1', 'conv3_2', 'conv3_3', 'conv3_4'], 'pool3', ['conv4_1', 'conv4_2', 'conv4_3', 'conv4_4'], 'pool4',
    ['conv5_1', 'conv5_2', 'conv5_3', 'conv5_4'], 'pool5', 'flatten', 'fc1_relu', 'fc2_relu', 'outputs'
]

if not _use_DeConv2d:
    layer_names_rev = [
        ['conv4_1'], 'upsample3',
        ['conv3_4', 'conv3_3', 'conv3_2', 'conv3_1'], 'upsample2',
        ['conv2_2', 'conv2_1'], 'upsample1',
        ['conv1_2', 'conv1_1']
    ]
else:
    layer_names_rev = [
        ['deconv4_1'], 'upsample3',
        ['deconv3_4', 'deconv3_3', 'deconv3_2', 'deconv3_1'], 'upsample2',
        ['deconv2_2', 'deconv2_1'], 'upsample1',
        ['deconv1_2', 'deconv1_1']
    ]

cfg = {  # NOTE: list type items are params for Conv2ds, containing filter numbers
    'A': [[64], 'M', [128], 'M', [256, 256], 'M', [512, 512], 'M', [512, 512], 'M', 'F', 'fc1', 'fc2', 'O'],
    'B': [[64, 64], 'M', [128, 128], 'M', [256, 256], 'M', [512, 512], 'M', [512, 512], 'M', 'F', 'fc1', 'fc2', 'O'],
    'D':
        [
            [64, 64], 'M', [128, 128], 'M', [256, 256, 256], 'M', [512, 512, 512], 'M', [512, 512, 512], 'M', 'F',
            'fc1', 'fc2', 'O'
        ],
    'E':  # conv1_1-2:[3,3,3,64],[64,64], conv2_1-2:[]
        [
            [64, 64], 'M', [128, 128], 'M', [256, 256, 256, 256], 'M', [512, 512, 512, 512], 'M', [512, 512, 512, 512],
            'M', 'F', 'fc1', 'fc2', 'O'
        ],
    'E_rev':  # conv4_1:[3,3,512,256], conv3_4-1:[3,3,256,256],[256,256],[256,256],[256,128], conv2_2-1:[3,3,128,128],[128,64], conv1_2-1:[3,3,64,64],[64,3]
        [
            [256], 'U', [256, 256, 256, 128], 'U', [128, 64], 'U', [64, 3]
        ],
}


mapped_cfg = {
    'vgg11': 'A',
    'vgg11_bn': 'A',
    'vgg13': 'B',
    'vgg13_bn': 'B',
    'vgg16': 'D',
    'vgg16_bn': 'D',
    'vgg19': 'E',
    'vgg19_bn': 'E',
    'vgg19_rev': 'E_rev',
}

model_urls = {
    'vgg16': 'http://www.cs.toronto.edu/~frossard/vgg16/',
    # 'vgg19': 'https://media.githubusercontent.com/media/tensorlayer/pretrained-models/master/models/'
    'vgg19': 'https://github.com/tensorlayer/pretrained-models/raw/master/models/'
}

model_saved_name = {'vgg16': 'vgg16_weights.npz', 'vgg19': 'vgg19.npy'}


# noinspection PyPep8Naming
class LayerList_before_merge(LayerList):
    def forward(self, inputs, observed_layer_names=None):
        observed_outputs = []
        z = inputs
        for layer in self.layers:
            z = layer.forward(z)
            if observed_layer_names is not None and layer.name in observed_layer_names:
                observed_outputs.append(z)
        if observed_layer_names is not None:
            return z, observed_outputs.copy()  # new version, return a tuple when outputs_layer_names specified
        else:
            return z  # old version, keep compatibility

# noinspection PyPep8Naming
class DeConv2d_before_merge(DeConv2d):
    def __init__(
        self,
        n_filter=32,
        filter_size=(3, 3),
        strides=(1, 1),
        act=None,
        padding='SAME',
        dilation_rate=(1, 1),
        data_format='channels_last',
        W_init=tl.initializers.truncated_normal(stddev=0.02),
        b_init=tl.initializers.constant(value=0.0),
        in_channels=None,
        name=None
    ):
        super(DeConv2d_before_merge, self).__init__(
            n_filter=n_filter, filter_size=filter_size, strides=strides, act=act, padding=padding,
            dilation_rate=dilation_rate,
            data_format=data_format, W_init=W_init, b_init=b_init, in_channels=in_channels, name=name)
        self.layer = None
        if self.in_channels:
            self.build(None)
            self._built = True

    def build(self, inputs_shape):
        self.layer = tf.keras.layers.Conv2DTranspose(
            filters=self.n_filter,
            kernel_size=self.filter_size,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate,
            activation=self.act,
            use_bias=(True if self.b_init is not None else False),
            kernel_initializer=self.W_init,
            bias_initializer=self.b_init,
            name=self.name
        )
        if inputs_shape is None:
            inputs_shape = [1, 1, 1, self.in_channels] if self.data_format=='channels_last' else [1, self.in_channels, 1, 1]
        _outputs = self.layer(
            tf.convert_to_tensor(np.random.uniform(size=inputs_shape), dtype=np.float32)
        )  #self.layer(np.random.uniform([1] + list(inputs_shape)))  # initialize weights
        assert(_outputs.shape is not None)
        self._trainable_weights = self.layer.weights

# noinspection PyPep8Naming
class BatchNorm2d_before_merged(BatchNorm):
    def _check_input_shape(self, inputs):
        # IMPROVE: move this to utils package
        def get_ndim(np_or_tf_obj):
            if isinstance(np_or_tf_obj, np.ndarray):
                return np_or_tf_obj.ndim
            if isinstance(np_or_tf_obj, tf.Tensor):
                return np_or_tf_obj.get_shape().ndims
            return None
        if get_ndim(inputs) != 4:
            raise ValueError('expected input to be 4D, but got {}D input'.format(inputs.ndim))


class VGG(Model):
    def __init__(self, layer_type, batch_norm=False, end_with='outputs', input_depth=3, name=None):
        super(VGG, self).__init__(name=name)
        self.is_reversed_model = layer_type.endswith('_rev')

        config = cfg[mapped_cfg[layer_type]]
        self.layers = make_layers(config, batch_norm, end_with,
                                  is_reversed=self.is_reversed_model, input_depth=input_depth)

    def forward(self, inputs: tf.Tensor, observed_layer_names=None):
        """
        inputs : tensor
            for normal VGG: Shape [None, height, width, 3], value range [0.0, 255.0].
            for reversed VGG: features which are output of a normal VGG.
        """
        if not self.is_reversed_model:
            if not isinstance(inputs, tf.Tensor):
                inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)
            elif inputs.dtype != tf.float32:  # e.g.tf.uint8
                inputs = tf.cast(inputs, tf.float32)
            inputs = inputs - np.array([123.68, 116.779, 103.939], dtype=np.float32).reshape([1, 1, 1, 3])
        else:
            pass  # for reversed model, inputs will be features, output of standard VGG model

        if observed_layer_names is None:
            out, observed_outputs = self.layers(inputs), None
        else:
            out, observed_outputs = self.layers(inputs, observed_layer_names=observed_layer_names)

        if self.is_reversed_model:
            out = out + np.array([123.68, 116.779, 103.939], dtype=np.float32).reshape([1, 1, 1, 3])
            out = tf.clip_by_value(out, 0.0, 255.0)

        return out if observed_layer_names is None else (out, observed_outputs)


def make_layers(config, batch_norm=False, end_with='outputs', is_reversed=False, input_depth=3):
    layer_list = []
    is_end = False
    for layer_group_idx, layer_group in enumerate(config):
        is_last_group = (layer_group_idx == len(config) - 1)
        if isinstance(layer_group, list):
            for idx, layer in enumerate(layer_group):  # NOTE: here 'layer' means n_filter of the Conv2d layer
                is_last_layer = is_last_group and (idx == len(layer_group) - 1)
                layer_name = layer_names[layer_group_idx][idx] if not is_reversed else layer_names_rev[layer_group_idx][idx]
                n_filter = layer
                if idx == 0:
                    if layer_group_idx > 0:
                        in_channels = config[layer_group_idx - 2][-1]
                    else:
                        # TL1to2: possibly be 1, or other depth value of input (e.g. features)
                        in_channels = input_depth
                else:
                    # TL1to2: should trace the previous n_filter
                    # in_channels = layer
                    in_channels = layer_group[idx-1]
                if not (is_reversed and _use_DeConv2d):
                    # NOTE: customized padding to avoid border artifacts
                    if _use_PadLayer_Reflect:
                        layer_list.append(PadLayer([[0, 0], [1, 1], [1, 1], [0, 0]], mode="REFLECT"))
                    layer_list.append(
                        Conv2d(
                            n_filter=n_filter, filter_size=(3, 3), strides=(1, 1),
                            padding='VALID' if _use_PadLayer_Reflect else 'SAME',
                            act=_relu_class if not batch_norm else None,
                            in_channels=in_channels, name=layer_name
                        )
                    )
                else:
                    # NOTE: DeConv2d is 1-N reverse mapping, cannot use Padding before calling DeConv2d
                    layer_list.append(
                        DeConv2d_before_merge(
                            n_filter=n_filter, filter_size=(3, 3), strides=(1, 1),
                            padding='SAME',
                            act=_relu_class if not batch_norm and not is_last_layer else None,
                            in_channels=in_channels, name=layer_name
                        )
                    )
                if batch_norm and not is_last_layer:
                    # layer_list.append(BatchNorm())
                    layer_list.append(BatchNorm2d_before_merged(num_features=n_filter, act=_relu_class))
                if layer_name == end_with:
                    is_end = True
                    break
        else:
            layer_name = layer_names[layer_group_idx] if not is_reversed else layer_names_rev[layer_group_idx]
            if layer_group == 'M':
                layer_list.append(MaxPool2d(filter_size=(2, 2), strides=(2, 2), padding='SAME', name=layer_name))
            elif layer_group == 'U':
                layer_list.append(UpSampling2d(scale=(2, 2), method='nearest', name=layer_name))
            elif layer_group == 'O':
                layer_list.append(Dense(n_units=1000, in_channels=4096, name=layer_name))
            elif layer_group == 'F':
                layer_list.append(Flatten(name='flatten'))
            elif layer_group == 'fc1':
                layer_list.append(Dense(n_units=4096, act=_relu_class, in_channels=512 * 7 * 7, name=layer_name))
            elif layer_group == 'fc2':
                layer_list.append(Dense(n_units=4096, act=_relu_class, in_channels=4096, name=layer_name))
            if layer_name == end_with:
                is_end = True
        if is_end:
            break
    # TL1to2: may use LayerList_before_merge, which overrode forward() to observe intermediate outputs
    if _use_LayerList_before_merge:
        tl.logging.warning('Using LayerList_before_merge... consider merging to master branch.')
        return LayerList_before_merge(layer_list)
    else:
        return LayerList(layer_list)


def restore_model(model, layer_type):
    logging.info("Restore pre-trained weights")
    # download weights
    maybe_download_and_extract(model_saved_name[layer_type], 'models', model_urls[layer_type])
    weights = []
    if layer_type == 'vgg16':
        npz = np.load(os.path.join('models', model_saved_name[layer_type]), allow_pickle=True)
        # get weight list
        for val in sorted(npz.items()):
            logging.info("  Loading weights %s in %s" % (str(val[1].shape), val[0]))
            weights.append(val[1])
            if len(model.all_weights) == len(weights):
                break
    elif layer_type == 'vgg19':
        npz = np.load(os.path.join('models', model_saved_name[layer_type]), allow_pickle=True, encoding='latin1').item()
        # get weight list
        for val in sorted(npz.items()):
            logging.info("  Loading %s in %s" % (str(val[1][0].shape), val[0]))
            logging.info("  Loading %s in %s" % (str(val[1][1].shape), val[0]))
            weights.extend(val[1])
            if len(model.all_weights) == len(weights):
                break
    else:
        raise TypeError(f'layer type not supported for restore_model(): {layer_type}')
    # assign weight values
    # UPDATE: weights must be shorter in len than model.all_weights (caller's duty to check)
    # assign_weights(weights, model)
    assign_weights(weights[:len(model.all_weights)], model)

    del weights


def VGG_static(layer_type, batch_norm=False, end_with='outputs', name=None):
    ni = Input([None, 224, 224, 3])
    n = Lambda(
        lambda x: x * 255 - np.array([123.68, 116.779, 103.939], dtype=np.float32).reshape([1, 1, 1, 3]), name='scale'
    )(ni)

    config = cfg[mapped_cfg[layer_type]]
    layers = make_layers(config, batch_norm, end_with)

    nn = layers(n)

    M = Model(inputs=ni, outputs=nn, name=name)
    return M


def vgg16(pretrained=False, batch_norm=False, end_with='outputs', mode='dynamic', name=None):
    """Pre-trained VGG16 model.

    Parameters
    ------------
    pretrained : boolean
        Whether to load pretrained weights. Default False.
    end_with : str
        The end point of the model. Default ``fc3_relu`` i.e. the whole model.
    mode : str.
        Model building mode, 'dynamic' or 'static'. Default 'dynamic'.
    name : None or str
        A unique layer name.

    Examples
    ---------
    Classify ImageNet classes with VGG16, see `tutorial_models_vgg.py <https://github.com/tensorlayer/tensorlayer/blob/master/example/tutorial_models_vgg.py>`__
    With TensorLayer

    >>> # get the whole model, without pre-trained VGG parameters
    >>> vgg = tl.models.vgg16()
    >>> # get the whole model, restore pre-trained VGG parameters
    >>> vgg = tl.models.vgg16(pretrained=True)
    >>> # use for inferencing
    >>> output = vgg(img, is_train=False)
    >>> probs = tf.nn.softmax(output)[0].numpy()

    Extract features with VGG16 and Train a classifier with 100 classes

    >>> # get VGG without the last layer
    >>> cnn = tl.models.vgg16(end_with='fc2_relu', mode='static').as_layer()
    >>> # add one more layer and build a new model
    >>> ni = Input([None, 224, 224, 3], name="inputs")
    >>> nn = cnn(ni)
    >>> nn = tl.layers.Dense(n_units=100, name='out')(nn)
    >>> model = tl.models.Model(inputs=ni, outputs=nn)
    >>> # train your own classifier (only update the last layer)
    >>> train_params = model.get_layer('out').trainable_weights

    Reuse model

    >>> # in dynamic model, we can directly use the same model
    >>> # in static model
    >>> vgg_layer = tl.models.vgg16().as_layer()
    >>> ni_1 = tl.layers.Input([None, 224, 244, 3])
    >>> ni_2 = tl.layers.Input([None, 224, 244, 3])
    >>> a_1 = vgg_layer(ni_1)
    >>> a_2 = vgg_layer(ni_2)
    >>> M = Model(inputs=[ni_1, ni_2], outputs=[a_1, a_2])

    """
    if mode == 'dynamic':
        model = VGG(layer_type='vgg16', batch_norm=batch_norm, end_with=end_with, name=name)
    elif mode == 'static':
        model = VGG_static(layer_type='vgg16', batch_norm=batch_norm, end_with=end_with, name=name)
    else:
        raise Exception("No such mode %s" % mode)
    if pretrained:
        restore_model(model, layer_type='vgg16')
    return model


def vgg19(pretrained=False, batch_norm=False, end_with='outputs', mode='dynamic', name=None):
    """Pre-trained VGG19 model.

    Parameters
    ------------
    pretrained : boolean
        Whether to load pretrained weights. Default False.
    end_with : str
        The end point of the model. Default ``fc3_relu`` i.e. the whole model.
    mode : str.
        Model building mode, 'dynamic' or 'static'. Default 'dynamic'.
    name : None or str
        A unique layer name.

    Examples
    ---------
    Classify ImageNet classes with VGG19, see `tutorial_models_vgg.py <https://github.com/tensorlayer/tensorlayer/blob/master/example/tutorial_models_vgg.py>`__
    With TensorLayer

    >>> # get the whole model, without pre-trained VGG parameters
    >>> vgg = tl.models.vgg19()
    >>> # get the whole model, restore pre-trained VGG parameters
    >>> vgg = tl.models.vgg19(pretrained=True)
    >>> # use for inferencing
    >>> output = vgg(img, is_train=False)
    >>> probs = tf.nn.softmax(output)[0].numpy()

    Extract features with VGG19 and Train a classifier with 100 classes

    >>> # get VGG without the last layer
    >>> cnn = tl.models.vgg19(end_with='fc2_relu', mode='static').as_layer()
    >>> # add one more layer and build a new model
    >>> ni = Input([None, 224, 224, 3], name="inputs")
    >>> nn = cnn(ni)
    >>> nn = tl.layers.Dense(n_units=100, name='out')(nn)
    >>> model = tl.models.Model(inputs=ni, outputs=nn)
    >>> # train your own classifier (only update the last layer)
    >>> train_params = model.get_layer('out').trainable_weights

    Reuse model

    >>> # in dynamic model, we can directly use the same model
    >>> # in static model
    >>> vgg_layer = tl.models.vgg19().as_layer()
    >>> ni_1 = tl.layers.Input([None, 224, 244, 3])
    >>> ni_2 = tl.layers.Input([None, 224, 244, 3])
    >>> a_1 = vgg_layer(ni_1)
    >>> a_2 = vgg_layer(ni_2)
    >>> M = Model(inputs=[ni_1, ni_2], outputs=[a_1, a_2])

    """
    if mode == 'dynamic':
        model = VGG(layer_type='vgg19', batch_norm=batch_norm, end_with=end_with, name=name)
    elif mode == 'static':
        model = VGG_static(layer_type='vgg19', batch_norm=batch_norm, end_with=end_with, name=name)
    else:
        raise Exception("No such mode %s" % mode)
    if pretrained:
        restore_model(model, layer_type='vgg19')
    return model


def vgg19_rev(pretrained=False, batch_norm=False, end_with='conv1_1', mode='dynamic', input_depth=3, name=None):
    """reversed version of Pre-trained VGG19 model, usually used as decoder (features -> input).

    Parameters
    ------------
    pretrained : boolean
        Whether to load pretrained weights. Default False.
    end_with : str
        The end point of the model. Default ``fc3_relu`` i.e. the whole model.
    mode : str.
        Model building mode, 'dynamic' or 'static'. Default 'dynamic'.
    input_depth : int
        Depth of input
    name : None or str
        A unique layer name.

    Examples
    ---------
    """
    if mode == 'dynamic':
        model = VGG(layer_type='vgg19_rev', batch_norm=batch_norm, end_with=end_with, input_depth=input_depth, name=name)
    elif mode == 'static':
        model = VGG_static(layer_type='vgg19_rev', batch_norm=batch_norm, end_with=end_with, name=name)
    else:
        raise Exception("No such mode %s" % mode)
    if pretrained:
        raise Exception("there's no pretrained version for vgg19_rev")
    return model


VGG16 = vgg16
VGG19 = vgg19

# models without pretrained parameters
# def vgg11(pretrained=False, end_with='outputs'):
#     model = VGG(layer_type='vgg11', batch_norm=False, end_with=end_with)
#     if pretrained:
#         model.restore_weights()
#     return model
#
#
# def vgg11_bn(pretrained=False, end_with='outputs'):
#     model = VGG(layer_type='vgg11_bn', batch_norm=True, end_with=end_with)
#     if pretrained:
#         model.restore_weights()
#     return model
#
#
# def vgg13(pretrained=False, end_with='outputs'):
#     model = VGG(layer_type='vgg13', batch_norm=False, end_with=end_with)
#     if pretrained:
#         model.restore_weights()
#     return model
#
#
# def vgg13_bn(pretrained=False, end_with='outputs'):
#     model = VGG(layer_type='vgg13_bn', batch_norm=True, end_with=end_with)
#     if pretrained:
#         model.restore_weights()
#     return model
#
#
# def vgg16_bn(pretrained=False, end_with='outputs'):
#     model = VGG(layer_type='vgg16_bn', batch_norm=True, end_with=end_with)
#     if pretrained:
#         model.restore_weights()
#     return model
#
#
# def vgg19_bn(pretrained=False, end_with='outputs'):
#     model = VGG(layer_type='vgg19_bn', batch_norm=True, end_with=end_with)
#     if pretrained:
#         model.restore_weights()
#     return model
