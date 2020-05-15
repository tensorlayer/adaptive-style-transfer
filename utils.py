from __future__ import print_function

import numpy as np
from os import listdir, remove
from os.path import join

# Temporary Wrapping
# from scipy.misc import imread, imresize
import cv2
def imread(path, mode='RGB'):
    # return cv2.imread(path, cv2.IMREAD_COLOR)
    return cv2.imdecode(np.fromfile(path,dtype=np.uint8), cv2.IMREAD_COLOR)
def imsave(path, image):
    return cv2.imwrite(path, image)
def imresize(image, dst_size: (list, tuple), interp='nearest'):
    """
    :param image:
    :param dst_size: [w, h]
    :param interp:
    :return:
    """
    if isinstance(dst_size, list):
        dst_size = tuple(dst_size)
    return cv2.resize(image, dst_size, interpolation=cv2.INTER_NEAREST)
def imresize_square(image, long_side: int, interp='nearest'):
    top, bottom, left, right = (0, 0, 0, 0)
    h, w, _ = image.shape
    longest_edge = max(h, w)
    if h < longest_edge:
        dh = longest_edge - h
        top = dh // 2
        bottom = dh - top
    elif w < longest_edge:
        dw = longest_edge - w
        left = dw // 2
        right = dw - left
    if h != w:
        background = [0, 0, 0]  # IMPROVE: use dominant color, or tf.pad(mode='REFLECT')
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=background)
    return cv2.resize(image, (long_side, long_side), interpolation=cv2.INTER_NEAREST)


def list_images(directory):
    images = []
    for file in listdir(directory):
        name = file.lower()
        if name.endswith('.png'):  # IMPROVE: use any() instead
            images.append(join(directory, file))
        elif name.endswith('.jpg'):
            images.append(join(directory, file))
        elif name.endswith('.jpeg'):
            images.append(join(directory, file))
    return images


def get_train_images(paths, resize_len=512, crop_height=256, crop_width=256):
    images = []
    for path in paths:
        image = imread(path, mode='RGB')
        if image is None:
            print(f"[WARN] Bypassed unreadable train image: {path}")
            continue
        height, width, _ = image.shape

        # UPDATE: images have same w but different h, must padding after imresize()
        # if height < width:
        #     new_height = resize_len
        #     new_width = int(width * new_height / height)
        # else:
        #     new_width = resize_len
        #     new_height = int(height * new_width / width)
        # image = imresize(image, [new_height, new_width], interp='nearest')
        image = imresize_square(image, resize_len, interp='nearest')
        new_width, new_height = resize_len, resize_len

        # crop the image
        start_h = np.random.choice(new_height - crop_height + 1)
        start_w = np.random.choice(new_width - crop_width + 1)
        image = image[start_h:(start_h + crop_height), start_w:(start_w + crop_width), :]

        images.append(image)

    if len(images) == 0:
        return []
    while len(images) < len(paths):  # bypassed something, so append duplications
        images.append(images[0])

    images = np.stack(images, axis=0)

    return images

def single_inputs_generator(paired_filenames, content_path, style_path, constrained_longer_side=1200):
    def constrained_resize(image):
        h, w, _ = image.shape
        if constrained_longer_side >= (h if h > w else w):
            return image
        h = constrained_longer_side if h > w else int(constrained_longer_side * h / w)
        w = constrained_longer_side if w >= h else int(constrained_longer_side * w / h)
        return imresize(image, [h, w], interp='nearest')
    for content_filename, style_filename in paired_filenames:
        try:
            content_image = constrained_resize(imread(join(content_path, content_filename), mode='RGB'))
            style_image = constrained_resize(imread(join(style_path, style_filename), mode='RGB'))
        except Exception as e:
            print(f'[ERROR] Failed reading test image: {e}')
            continue  # bypass
        yield [content_image], [style_image]

# NOTE: this computation will also be tracked by AutoGraph
# Normalizes the `content_features` with scaling and offset from `style_features`.
def AdaIN(content_features, style_features, alpha=1, epsilon=1e-5):
    import tensorflow as tf
    # UPDATE: keep_dims -> keepdims
    content_mean, content_variance = tf.nn.moments(content_features, [1, 2], keepdims=True)
    style_mean, style_variance = tf.nn.moments(style_features, [1, 2], keepdims=True)

    normalized_content_features = tf.nn.batch_normalization(
        content_features, content_mean, content_variance, style_mean, tf.sqrt(style_variance), epsilon
    )
    normalized_content_features = alpha * normalized_content_features + (1 - alpha) * content_features
    return normalized_content_features


def pre_process_dataset(dir_path, shorter_side=512):
    import tensorlayer as tl

    paths = tl.files.load_file_list(dir_path, regx='\\.(jpg|jpeg|png)', keep_prefix=True)

    print('\norigin files number: %d\n' % len(paths))

    num_delete = 0

    for path in paths:

        try:
            image = imread(path, mode='RGB')
        except IOError:
            num_delete += 1
            print('Cant read this file, will delete it')
            remove(path)

        if len(image.shape) != 3 or image.shape[2] != 3:
            num_delete += 1
            remove(path)
            print('\nimage.shape:', image.shape, ' Remove image <%s>\n' % path)
        else:
            height, width, _ = image.shape

            if height < width:
                new_height = shorter_side
                new_width = int(width * new_height / height)
            else:
                new_width = shorter_side
                new_height = int(height * new_width / width)

            try:
                image = imresize(image, [new_height, new_width], interp='nearest')
            except Exception():
                print('Cant resize this file, will delete it')
                num_delete += 1
                remove(path)

    print('\n\ndelete %d files! Current number of files: %d\n\n' % (num_delete, len(paths) - num_delete))
