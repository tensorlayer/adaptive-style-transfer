import utils
import os.path as osp

def test_utils_imoperations():
    from utils import imread, imresize_square, get_train_images, imsave
    path_read = '/tmp/panda.jpg'
    path_save = '/tmp/panda_resized.jpg'
    # image = imread(path_read, mode='RGB')
    # image = imresize_square(image, long_side=256, interp = 'nearest')
    # imsave(path_save, image)
    images = get_train_images([path_read])
    imsave(path_save, images[0])


# noinspection PyPep8Naming
def test_test_arbitrary_sized_inputs():
    from vgg import vgg19, vgg19_rev
    import os.path as osp
    import tensorlayer as tl
    DEC_LATEST_WEIGHTS_PATH = 'pretrained_models/dec_latest_weights.h5'
    STYLE_LAYERS = ('conv1_1', 'conv2_1', 'conv3_1', 'conv4_1')  # for Encoders
    CONTENT_DATA_PATH = './dataset/content_samples'  # COCO_train_2014/'
    STYLE_DATA_PATH = './dataset/style_samples'  # wiki_all_images/'
    test_content_filenames = ['000000532397.jpg']  #, '000000048289.jpg', '000000526781.jpg']
    test_style_filenames = ['53154.jpg']  #, '2821.jpg', '216.jpg']
    TEST_INPUT_CONSTRAINTED_SIZE = 800
    TEMP_IMAGE_PATH = './temp_images/'

    tl.logging.set_verbosity(tl.logging.DEBUG)
    enc_net = vgg19(pretrained=True, end_with='conv4_1')
    # NOTE: batch_norm=True will lower quality of the generated image = need retrain
    dec_net = vgg19_rev(pretrained=False, batch_norm=False, input_depth=512)
    if osp.exists(DEC_LATEST_WEIGHTS_PATH):
        dec_net.load_weights(DEC_LATEST_WEIGHTS_PATH, skip=True)

    enc_net.eval()
    dec_net.eval()
    for epoch in range(1):  # for test generator validity
        # Note: generator need reset for reuse
        test_inputs_gen = utils.single_inputs_generator(list(zip(test_content_filenames, test_style_filenames)),
                                                        CONTENT_DATA_PATH, STYLE_DATA_PATH, TEST_INPUT_CONSTRAINTED_SIZE)
        for i, (test_content, test_style) in enumerate(test_inputs_gen):
            # shape=[1, w, h, c], so as to feed arbitrary sized test images one by one
            content_features = enc_net(test_content)
            style_features = enc_net(test_style,)
            target_features = utils.AdaIN(content_features, style_features, alpha=1)
            del content_features, style_features
            generated_images = dec_net(target_features)
            paired_name = f"{osp.splitext(test_style_filenames[i])[0]}+{osp.splitext(test_content_filenames[i])[0]}"
            utils.imsave(osp.join(TEMP_IMAGE_PATH, f"temp_{paired_name}_epoch{epoch}.jpg"), generated_images[0].numpy())

def test_vgg19_save_weights():
    from vgg import vgg19
    MODEL_SAVE_PATH = './trained_models/'
    enc_c_net = vgg19(pretrained=True, end_with='conv4_1', name='content')
    enc_c_net.save_weights(MODEL_SAVE_PATH + 'predefined_vgg19_endwith(conv4_1)_weights.h5')

def test_conv_and_deconv():
    VGG19_WEIGHTS_PATH = 'pretrained_models/predefined_vgg19_endwith(conv4_1)_weights.h5'
    VGG19_REV_WEIGHTS_PATH = 'pretrained_models/dec_best_weights (before use DeConv2d).h5'
    TEMP_IMAGE_PATH = './temp_images/53154.jpg'
    # try directly decoding content features
    enc_net = vgg19(pretrained=False, end_with='conv4_1')
    dec_net = vgg19_rev(pretrained=False, end_with='conv1_1', input_depth=512)
    enc_net.load_weights(VGG19_WEIGHTS_PATH)
    dec_net.load_weights(VGG19_REV_WEIGHTS_PATH, skip=True)
    enc_net.eval()
    dec_net.eval()
    image = imread(TEMP_IMAGE_PATH, mode='RGB')
    image = imresize_square(image, long_side=512, interp='nearest')
    content_features = enc_net([image])
    generated_images = dec_net(content_features)
    imsave(TEMP_IMAGE_PATH + '!generated.jpg', generated_images[0].numpy())

def test_vgg19_rev_save_weights():
    from vgg import vgg19_rev
    MODEL_SAVE_PATH = './trained_models/'
    dec_c_net = vgg19_rev(pretrained=False, end_with='conv1_1', input_depth=512, name='stylized_dec')
    dec_c_net.save_weights(osp.join(MODEL_SAVE_PATH, 'temp_vgg19_rev_weights.h5'))

def test_vgg19_rev_load_weights():
    from vgg import vgg19_rev
    DEC_LATEST_WEIGHTS_PATH = 'pretrained_models/dec_latest_weights.h5'
    tl.logging.set_verbosity(tl.logging.DEBUG)
    dec_c_net = vgg19_rev(pretrained=False, batch_norm=True, end_with='conv1_1', input_depth=512, name='stylized_dec')
    dec_c_net.load_weights(DEC_LATEST_WEIGHTS_PATH, skip=True)

def test_interrupt_event():
    import time
    import logging
    try:
        i = 0
        while True:
            logging.info(f"waiting for ctrl+c...: {i+1:8d}")
            time.sleep(5)  # seconds
            i += 1
    except KeyboardInterrupt:
        logging.info(f"ctrl+c event caught.")

def test_vgg_rev_load_vgg_weights():
    from vgg import vgg19_rev
    VGG19_WEIGHTS_PATH = 'pretrained_models/predefined_vgg19_endwith(conv4_1)_weights.h5'
    TEMP_IMAGE_PATH = './temp_images/'
    # enc_c_net = vgg19(pretrained=False, end_with='conv4_1', name='content_enc')
    # enc_c_net.load_weights(VGG19_WEIGHTS_PATH)
    dec_c_net = vgg19_rev(pretrained=False, end_with='conv1_1', input_depth=512, name='content_dec')
    dec_c_net.load_weights(VGG19_WEIGHTS_PATH, skip=True)


