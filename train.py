from datetime import datetime

import os.path as osp

import numpy as np
# from scipy.misc import imsave
from utils import imsave
import tensorlayer as tl
from tensorlayer.models import Model
import utils

# TL1to2: self-defined VGG-alike models -> reuse models\vgg.py
# from models import Encoder, Decoder
from vgg import vgg19, vgg19_rev

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

VGG19_PARTIAL_WEIGHTS_PATH = 'pretrained_models/predefined_vgg19_endwith(conv4_1)_weights.h5'
DEC_LATEST_WEIGHTS_PATH = 'pretrained_models/dec_latest_weights.h5'
STYLE_LAYERS = ('conv1_1', 'conv2_1', 'conv3_1', 'conv4_1')  # for Encoders
STYLE_WEIGHT = 2.0  # affects value of the loss

EPOCHS = 960
EPSILON = 1e-5
USE_BATCH_NORM = False
LEARNING_RATE = 1e-3 if USE_BATCH_NORM else 1e-4       # 1e-3 performs bitterly worse, if bn is off
BATCH_SIZE = 8
HEIGHT = 256
WIDTH = 256
CHANNEL = 3
INPUT_SHAPE = (BATCH_SIZE, HEIGHT, WIDTH, CHANNEL)

CONTENT_DATA_PATH = './dataset/content_samples'  # COCO_train_2014/'
STYLE_DATA_PATH = './dataset/style_samples'  # wiki_all_images/'
test_content_filenames = ['000000532397.jpg', '000000048289.jpg', '000000526781.jpg']
test_style_filenames = ['53154.jpg', '2821.jpg', '216_01.jpg']
TEST_INPUT_CONSTRAINTED_SIZE = 800
MODEL_SAVE_PATH = './trained_models/'
TEMP_IMAGE_PATH = './temp_images/'

# NOTE: If you have imported Scipy or alike, Interrupt Handler might have been injected.
def adjust_interrupt_handlers():
    import os
    import imp
    import ctypes
    import _thread
    import win32api

    # Load the DLL manually to ensure its handler gets
    # set before our handler.
    basepath = imp.find_module('numpy')[1]
    def try_to_load(dll_path):
        try:
            ctypes.CDLL(dll_path)
        except OSError as e:
            pass
    try_to_load(os.path.join(basepath, 'core', 'libmmd.dll'))
    try_to_load(os.path.join(basepath, 'core', 'libifcoremd.dll'))

    # Now set our handler for CTRL_C_EVENT. Other control event 
    # types will chain to the next handler.
    def handler(dwCtrlType, hook_sigint=_thread.interrupt_main):
        if dwCtrlType == 0: # CTRL_C_EVENT
            hook_sigint()
            return 1 # don't chain to the next handler
        return 0 # chain to the next handler

    win32api.SetConsoleCtrlHandler(handler, 1)

if __name__ == '__main__':

    tl.logging.set_verbosity(tl.logging.INFO)  # TEMP: DEBUG, INFO

    adjust_interrupt_handlers()  # for handling Ctrl+C interrupt
    start_time = datetime.now()

    # IMPROVE: may use generator and tf.data.Dataset.from_generator + map_fn, which supports shuffle, batch etc.
    # Get the path of all valid images
    print('Preprocessing training images \n')
    tl.logging.info("Preprocessing training images")
    content_image_paths = tl.files.load_file_list(CONTENT_DATA_PATH, regx='\\.(jpg|jpeg|png)', keep_prefix=True)
    style_image_paths = tl.files.load_file_list(STYLE_DATA_PATH, regx='\\.(jpg|jpeg|png)', keep_prefix=True)
    num_imgs = min(len(content_image_paths), len(style_image_paths))
    content_image_paths = content_image_paths[:num_imgs]
    style_image_paths = style_image_paths[:num_imgs]
    mod = num_imgs % BATCH_SIZE
    print('Preprocessing finish, %d images in total \n' % (num_imgs - mod))
    if mod > 0:
        print('Train set has been trimmed %d samples...\n' % mod)
        content_image_paths = content_image_paths[:-mod]
        style_image_paths = style_image_paths[:-mod]

    # TL1to2: tf.Session -> obsolete
    # with tf.Graph().as_default(), tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:

    # TL1to2: Encode/Decode NN -> move to instance attributes of Model class
    # encoder = Encoder()
    # decoder = Decoder()

    # TL1to2: Input -> will be directly fed to Model.__call__
    # content_input = tf.placeholder(tf.float32, shape=INPUT_SHAPE, name='content_input')
    # style_input = tf.placeholder(tf.float32, shape=INPUT_SHAPE, name='style_input')

    import tensorflow as tf

    # TL1to2: dynamic modeling, will take Input tensors as params
    class StyleTransferModel(Model):
        def __init__(self, style_weight = STYLE_WEIGHT):
            super(StyleTransferModel, self).__init__()
            # NOTE: you may check on `pretrained` if you want to download complete version of vgg19 weights
            want_to_download_vgg19 = False
            self.enc_net = vgg19(pretrained=want_to_download_vgg19, end_with='conv4_1', name='content_and_style_enc')
            if not want_to_download_vgg19 and osp.exists(VGG19_PARTIAL_WEIGHTS_PATH):
                self.enc_net.load_weights(VGG19_PARTIAL_WEIGHTS_PATH)
                tl.logging.info(f"Encoder weights loaded from: {VGG19_PARTIAL_WEIGHTS_PATH}")
            # NOTE: batch_norm=False->True will lower quality of the generated image = may need retrain
            self.dec_net = vgg19_rev(pretrained=False, batch_norm=USE_BATCH_NORM, end_with='conv1_1', input_depth=512, name='stylized_dec')
            if osp.exists(DEC_LATEST_WEIGHTS_PATH):
                self.dec_net.load_weights(DEC_LATEST_WEIGHTS_PATH, skip=True)
                tl.logging.info(f"Decoder weights loaded from: {DEC_LATEST_WEIGHTS_PATH}")
            self.style_weight = style_weight
            self.content_loss, self.style_loss, self.loss = None, None, None

        def forward(self, inputs: list, training=None, alpha=1):
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

            # 1.encode image: get content features and style features (i.e. intermediate style features)
            c_content_features = self.enc_net(content)
            s_content_features, s_style_feats_in_layers = self.enc_net(style, observed_layer_names=STYLE_LAYERS)

            # 2.pass the encoded content and style to AdaIN
            target_features = utils.AdaIN(c_content_features, s_content_features, alpha=alpha)

            # 3.decode target features back to generate an image
            generated_images = self.dec_net(target_features)

            # # de-preprocess image
            # generated_images = Encoder.reverse_preprocess(generated_images)
            # # switch BGR back to RGB
            # generated_images = tf.reverse(generated_images, axis=[-1])
            # # clip to 0..255
            # generated_images = tf.clip_by_value(generated_images, 0.0, 255.0)

            # 4.compute content features and style features of the generated image
            g_content_features, g_style_feats_in_layers = self.enc_net(generated_images, observed_layer_names=STYLE_LAYERS)
            tl.logging.info(
                f"c_c_feat:{c_content_features.shape}, s_c_feat:{s_content_features.shape}, "
                f"t_feat:{target_features.shape}, g:{generated_images.shape}, g_c_feat:{g_content_features.shape}")

            # 5.compute losses
            self.content_loss = tf.reduce_sum(
                tf.reduce_mean(tf.square(g_content_features - target_features), axis=[1, 2]))

            style_layer_loss = []
            for idx, layer_name in enumerate(STYLE_LAYERS):
                # TL1to2: tl.layers.get_layers_with_name -> observe intermediate outputs through model.__call__
                # s_style_feat = tl.layers.get_layers_with_name(self.enc_s_net, 'style/' + layer, True)[0]
                # g_style_feat = tl.layers.get_layers_with_name(self.enc_net, 'stylized_enc/' + layer, True)[0]
                s_style_feat = s_style_feats_in_layers[idx]
                g_style_feat = g_style_feats_in_layers[idx]
                mean_s, var_s = tf.nn.moments(s_style_feat, [1, 2])
                mean_g, var_g = tf.nn.moments(g_style_feat, [1, 2])
                sigma_s = tf.sqrt(var_s + EPSILON)
                sigma_g = tf.sqrt(var_g + EPSILON)
                l2_mean = tf.reduce_sum(tf.square(mean_g - mean_s))
                l2_sigma = tf.reduce_sum(tf.square(sigma_g - sigma_s))
                style_layer_loss.append(l2_mean + l2_sigma)
            self.style_loss = tf.reduce_sum(style_layer_loss)

            self.loss = self.content_loss + self.style_weight * self.style_loss  # IMPROVE: tune STYLE_WEIGHT

            return generated_images

    style_transfer_model = StyleTransferModel()
    opt = tf.optimizers.Adam(learning_rate=LEARNING_RATE)

    # Training step (Only train the decoder params)
    # TL1to2: train step -> tf.function
    # train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss, var_list=stylized_dec_net.all_params)
    @tf.function
    def train_step(inputs: list):
        with tf.GradientTape() as tape:
            generated_img = style_transfer_model(inputs, is_train=True)
        grad = tape.gradient(style_transfer_model.loss, style_transfer_model.dec_net.trainable_weights)
        opt.apply_gradients(zip(grad, style_transfer_model.dec_net.trainable_weights))
        return style_transfer_model.content_loss, style_transfer_model.style_loss, style_transfer_model.loss


    # TL1to2: session-> obsolete
    # sess.run(tf.global_variables_initializer())

    # TL1to2: restore vgg weights from .npz -> use pretrained model vgg, in StyleTransferModel
    # encoder.restore_model(sess, ENCODER_WEIGHTS_PATH, enc_c_net)
    # encoder.restore_model(sess, ENCODER_WEIGHTS_PATH, enc_s_net)
    # encoder.restore_model(sess, ENCODER_WEIGHTS_PATH, stylized_enc_net)

    # """Start Training"""
    step, is_last_step = 0, False
    best_loss = None
    result_images = []
    n_batches = int(num_imgs // BATCH_SIZE)
    _content_loss, _style_loss, _loss = None, None, None

    elapsed_time = datetime.now() - start_time
    print('\nElapsed time for preprocessing before actually train the model: %s' % elapsed_time)

    print('Now begin to train the model...\n')
    start_time = datetime.now()

    try:
        for epoch in range(EPOCHS):

            np.random.shuffle(content_image_paths)
            np.random.shuffle(style_image_paths)

            for batch in range(n_batches):

                is_last_step = (epoch == EPOCHS-1) and (batch == n_batches-1)

                # retrive a batch of content and style images
                content_batch_paths = content_image_paths[batch * BATCH_SIZE:(batch + 1) * BATCH_SIZE]
                style_batch_paths = style_image_paths[batch * BATCH_SIZE:(batch + 1) * BATCH_SIZE]

                content_batch = utils.get_train_images(content_batch_paths, crop_height=HEIGHT, crop_width=WIDTH)
                style_batch = utils.get_train_images(style_batch_paths, crop_height=HEIGHT, crop_width=WIDTH)

                # IMPROVE: preprocess dataset to prevent reading failure and retires
                if len(content_batch) < BATCH_SIZE or len(style_batch) < BATCH_SIZE:
                    continue  # bypass this batch...

                # run the training step
                # TL1to2: session-> obsolete
                # sess.run(train_op, feed_dict={content_input: content_batch, style_input: style_batch})
                _content_loss, _style_loss, _loss = train_step([content_batch, style_batch])

                if step > 0 and step % 100 == 0 or is_last_step:
                    # TL1to2: session-> obsolete
                    # _content_loss, _style_loss, _loss = sess.run(
                    #     [content_loss, style_loss, loss], feed_dict={
                    #         content_input: content_batch,
                    #         style_input: style_batch
                    #     }
                    # )
                    # IMPROVE: collect and plot metrics (tf.summary and TensorBoard)
                    elapsed_time = datetime.now() - start_time
                    print('step: %d,  total loss: %.3f, elapsed time: %s' % (step, _loss, elapsed_time))
                    print('content loss: %.3f' % _content_loss)
                    print(
                        'style loss  : %.3f,  weighted style loss: %.3f\n' % (_style_loss, STYLE_WEIGHT * _style_loss)
                    )
                    if step >= 1000 and (best_loss is None or best_loss > _loss):
                        # TL1to2: weights save/lod -> use save_weights/load_weights
                        print('save model weights now,step:', step)
                        # tl.files.save_npz(stylized_dec_net.all_params, name=MODEL_SAVE_PATH + str(step) + '_model.npz')
                        style_transfer_model.dec_net.save_weights(MODEL_SAVE_PATH + f'dec_{step}(loss={_loss:.2f})_weights.h5')
                        best_loss = _loss

                if step > 0 and step % 1000 == 0 or is_last_step:
                    # result_image = sess.run(
                    #     generated_img, feed_dict={
                    #         content_input: content_batch,
                    #         style_input: style_batch
                    #     }
                    # )
                    test_inputs_gen = utils.single_inputs_generator(list(zip(test_content_filenames, test_style_filenames)),
                                                                    CONTENT_DATA_PATH, STYLE_DATA_PATH, TEST_INPUT_CONSTRAINTED_SIZE)
                    for i, (test_contents, test_styles) in enumerate(test_inputs_gen):
                        # shape=[1, w, h, c] for contents and styles, so as to feed arbitrary sized test samples
                        paired_name = f"{osp.splitext(test_style_filenames[i])[0]}" \
                                      f"+{osp.splitext(test_content_filenames[i])[0]}"
                        try:
                            # IMPROVE: tune alpha. a value smaller than 1.0 will keep more content and convert less style
                            result_images = style_transfer_model([test_contents, test_styles], is_train=False, alpha=1)
                            print(f"generated_img for test ({paired_name}): {result_images[0].shape}")
                            utils.imsave(osp.join(TEMP_IMAGE_PATH, f"{paired_name}_{step}.jpg"), result_images[0].numpy())
                        except Exception as e:
                            tl.logging.error(f"failed to encode or save test image, bypassed: {paired_name}")

                if not is_last_step:
                    step += 1

            print(f'One Epoch finished! ({step}steps)\n' if not is_last_step else f'All Epochs finished! ({step}steps)\n')

        # """Done Training & Save the model"""
        # TL1to2: weights save/lod -> use save_weights/load_weights
        # tl.files.save_npz(stylized_dec_net.all_params, name=MODEL_SAVE_PATH + str(step) + '_model.npz')
        # ... move into the loop, ref: is_last_step
    except KeyboardInterrupt:
        print('Interrupted by keyboard.')
        try:
            print('save model weights for latest step:', step)
            style_transfer_model.dec_net.save_weights(MODEL_SAVE_PATH + f'dec_{step}(loss={_loss:.2f})_weights.h5')
            print('saved.')
        except Exception as e:
            pass

