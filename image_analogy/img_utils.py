import numpy as np
from imageio import imread
from PIL import Image

from . import vgg16


# util function to open, resize and format pictures into appropriate tensors
def load_image(image_path):
    return imread(image_path) # , mode='RGB')  # NOTE: this mode kwarg requires v0.17


def imresize(img, size, interp):
    resample = 0
    if interp == 'nearest':
        resample = Image.NEAREST
    elif interp == 'lanczos':
        resample = Image.LANCZOS
    elif interp == 'bilinear':
        resample = Image.BILINEAR
    elif interp == 'bicubic':
        resample = Image.BICUBIC
    elif interp == 'cubic':
        resample = Image.CUBIC

    return np.asarray(Image.fromarray(img).resize((size[1], size[0]), resample))



# util function to open, resize and format pictures into appropriate tensors
def preprocess_image(x, img_width, img_height):
    img = imresize(x, (img_height, img_width), interp='bicubic').astype(np.float32)
    img = vgg16.img_to_vgg(img)
    img = np.expand_dims(img, axis=0)
    return img


# util function to convert a tensor into a valid image
def deprocess_image(x, contrast_percent=0.0, resize=None):
    x = vgg16.img_from_vgg(x)
    if contrast_percent:
        min_x, max_x = np.percentile(x, (contrast_percent, 100 - contrast_percent))
        x = (x - min_x) * 255.0 / (max_x - min_x)
    x = np.clip(x, 0, 255)
    if resize:
        x = imresize((x * 255).astype(np.uint8), resize, interp='bicubic')
    return x.astype('uint8')


def reshape_weights(weights):
    kernel = np.asarray(weights)
    if not 3 <= kernel.ndim <= 5:
        raise ValueError('Invalid kernel shape:', kernel.shape)
    slices = [slice(None, None, -1) for _ in range(kernel.ndim)]
    no_flip = (slice(None, None), slice(None, None))
    slices[-2:] = no_flip
    return np.copy(kernel[tuple(slices)]).transpose((2, 3, 1, 0)) # use `arr[tuple(seq)]` instead of `arr[seq]
    return weights
