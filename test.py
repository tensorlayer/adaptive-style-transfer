from datetime import datetime
import os
import numpy as np

import tensorflow as tf
# from tensorlayer.layers import Layer, Input, Dropout, Dense
from tensorlayer.models import Model

# from models import Decoder, Encoder
from vgg import vgg19, vgg19_rev

# from scipy.misc import imread, imsave
from utils import imread, imsave
import utils

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# TL1to2: self-defined VGG-alike models -> reuse pretrained models\vgg.py
# ENCODER_WEIGHTS_PATH = 'pretrained_models/pretrained_vgg19_encoder_weights.npz'
# DECODER_WEIGHTS_PATH = 'pretrained_models/pretrained_vgg19_decoder_weights.npz'
VGG19_PARTIAL_WEIGHTS_PATH = 'pretrained_models/predefined_vgg19_endwith(conv4_1)_weights.h5'
DEC_BEST_WEIGHTS_PATH = 'pretrained_models/dec_best_weights.h5'

content_path = 'test_images/content/'
style_path = 'test_images/style/'
output_path = 'test_images/output/'

if __name__ == '__main__':

    content_image_paths = os.listdir(content_path)
    style_image_paths = os.listdir(style_path)

    # TL1to2: Encode/Decode NN -> take as instance attributes of Model class
    # encoder = Encoder()
    # decoder = Decoder()

    # TL1to2: Input -> directly feed to callable Model
    # content_input = tf1.placeholder(tf.float32, shape=(1, None, None, 3), name='content_input')
    # style_input = tf1.placeholder(tf.float32, shape=(1, None, None, 3), name='style_input')

    # TL1to2: dynamic modeling, will take Input tensors as params
    class StyleTransferModel(Model):
        def __init__(self, *args, **kwargs):
            super(StyleTransferModel, self).__init__(*args, **kwargs)
            # NOTE: you may use a vgg19 instance for both content encoder and style encoder, just as in train.py
            # self.enc_c_net = vgg19(pretrained=True, end_with='conv4_1', name='content')
            # self.enc_s_net = vgg19(pretrained=True, end_with='conv4_1', name='style')
            self.enc_net = vgg19(pretrained=False, end_with='conv4_1', name='content_and_style_enc')
            if os.path.exists(VGG19_PARTIAL_WEIGHTS_PATH):
                self.enc_net.load_weights(VGG19_PARTIAL_WEIGHTS_PATH, in_order=False)
            self.dec_net = vgg19_rev(pretrained=False, end_with='conv1_1', input_depth=512, name='stylized_dec')
            if os.path.exists(DEC_BEST_WEIGHTS_PATH):
                self.dec_net.load_weights(DEC_BEST_WEIGHTS_PATH, skip=True)

        def forward(self, inputs, training=None, alpha=1):
            """
            :param inputs: [content_batch, style_batch], both have shape as [batch_size, w, h, c]
            :param training:
            :param alpha:
            :return:
            """
            # TL1to2: preprocessing and reverse -> vgg forward() will handle it
            # # switch RGB to BGR
            # content = tf.reverse(content_input, axis=[-1])
            # style = tf.reverse(style_input, axis=[-1])
            # # preprocess image
            # content = Encoder.preprocess(content_input)
            # style = Encoder.preprocess(style_input)
            content, style = inputs

            # encode image
            # we should initial global variables before restore model
            content_features = self.enc_net(content)
            style_features = self.enc_net(style)

            # pass the encoded images to AdaIN  # IMPROVE: try alpha gradients
            target_features = utils.AdaIN(content_features, style_features, alpha=alpha)

            # decode target features back to image
            generated_img = self.dec_net(target_features)

            # # deprocess image
            # generated_img = Encoder.reverse_preprocess(generated_img)
            # # switch BGR back to RGB
            # generated_img = tf.reverse(generated_img, axis=[-1])
            # # clip to 0..255
            # generated_img = tf.clip_by_value(generated_img, 0.0, 255.0)

            return generated_img

    style_transfer_model = StyleTransferModel()

    start_time = datetime.now()
    image_count = 0
    for s_path in style_image_paths:
        # Load image from path and add one extra dimension to it.
        style_images = [imread(os.path.join(style_path, s_path), output_mode='RGB')]

        for c_path in content_image_paths:
            content_images = [imread(os.path.join(content_path, c_path), output_mode='RGB')]

            # TL1to2: session -> obsolete
            # result = sess.run(generated_img, feed_dict={content_input: content_tensor, style_input: style_tensor})
            # IMPROVE: tune alpha. a value smaller than 1.0 will keep more content and convert less style
            result = style_transfer_model([content_images, style_images], is_train=False, alpha=1)
            del content_images

            result_path = os.path.join(output_path, c_path.split('.')[0] + '_' + s_path.split('.')[0] + '.jpg')
            imsave(result_path, result[0].numpy())
            # tl.vis.save_image(result[0].transpose((1, 0, 2)), result_path) # [w,h,c_path] required
            print(result_path, ' is generated')
            del result
            image_count = image_count + 1
    elapsed_time = datetime.now() - start_time
    print("total image:", image_count, " total_time ", elapsed_time, " average time:", elapsed_time / image_count)
