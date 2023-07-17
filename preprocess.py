# Some functions here from haiku/examples/imagenet/dataset.py

from random import random
import keras_cv
import tensorflow as tf
layers = tf.keras.layers
from official.vision.ops import augment
from typing import Optional, Tuple
from jax import numpy as jnp
import jax



cropSizes = [116, 230, 460, 920]
cropSize = -1


def deconstruct(x):
    return x['image'], x['label']

def reconstruct(x,y):
    return {'image': x, 'label': y}

def cast_fn(x):
    # batch = dict(**batch)
    x = tf.cast(x, tf.dtypes.as_dtype(tf.float32))
    return x

def cast_as_uint8(x):
    x = tf.cast(x, tf.dtypes.as_dtype(tf.uint8))
    return x

#No longer used in favor if on-GPU version
def normalize(x):
    STDDEV_RGB = tf.convert_to_tensor(42.48881573)
    MEAN_RGB = tf.convert_to_tensor(126.52482573)
    x -= MEAN_RGB
    x = tf.math.divide(x, STDDEV_RGB)
    return x

# This was used with tfds; no longer used in favor of JAX on-GPU version
def randomAugments(x):
    #Want an overall rate of 0.8.
    #0.9^2, ok
    #x^n = 0.8
    #x = 0.8^(1/n)
    p = 0.9**(1/7)
    if random() > p:
        x = augment.autocontrast(x)
    if random() > p:
        x = augment.equalize(x)
    if random() > p:
        x = augment.invert(x)
    if random() > p:
        bits = int(random()*4)+4
        x = augment.posterize(x, bits)
    if random() > p:
        factor = int(random())
        x = augment.color(x, factor)
    if random() > p:
        factor = int(random())/4 +0.75
        x = augment.sharpness(x, factor)
    #
    if random() > p:
        size = get_cutout_size(x)
        var = random()*0.4+0.8
        size = int(size*var)
        x = augment.cutout(x, size, 128)
    #
    if random() > p:
        threshold = int(random() * 64) + 128
        addition = int(random()*256)-128
        x = augment.solarize_add(x, addition=addition,
                                 threshold=threshold)
    return x

# This was discovered after building the above function, and is comparable
randAugment = keras_cv.layers.RandAugment(value_range=(0,255))

# This function was used before moving to RandAugment
def keras_image_augmentation(x):
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        # layers.RandomRotation(0.125),
        # layers.RandomTranslation(0.2, 0.2),
        # layers.RandomBrightness(0.2),
    ])(x)

def rgb(x):
    return tf.image.grayscale_to_rgb(x)


greyscale = keras_cv.layers.Grayscale(output_channels=1)

def preprocess(x):
    # This is a little barebones; before moving preprocessing to GPU,
    # everything was done here via tfds on cpu.
    x,y = deconstruct(x)
    if x.get_shape()[2] == 1:
        x = rgb(x)
    x = cast_fn(x)
    return reconstruct(x, y)


def get_cutout_size(image):
    shape = image.get_shape()
    if shape[0] != shape[1]:
        print(shape)
        raise(BaseException())

    if shape[0] == 128:
        return shape[0]-cropSizes[0]
    elif shape[0] == 256:
        return shape[0]-cropSizes[1]
    elif shape[0] == 512:
        return shape[0]-cropSizes[2]
    elif shape[0] == 1024:
        return shape[0]-cropSizes[3]
    else:
        return shape[0] - int(float(shape[0]) * 0.9)

def _get_crop_size(image_bytes):
    jpeg_shape = tf.image.extract_jpeg_shape(image_bytes)
    # if jpeg_shape[0] != jpeg_shape[1]:
    #     raise ValueError('Non-square images are not currently supported. Height: {}, width: {}'
    #                      .format(jpeg_shape[0], jpeg_shape[1]))
    if jpeg_shape[0] == 128:
        return cropSizes[0]
    elif jpeg_shape[0] == 256:
        return cropSizes[1]
    elif jpeg_shape[0] == 512:
        return cropSizes[2]
    elif jpeg_shape[0] == 1024:
        return cropSizes[3]
    else:
        return int(float(jpeg_shape[0]) * 0.9)


def _decode_and_random_crop(image_bytes: tf.Tensor) -> tf.Tensor:
    """Make a random crop of 224."""
    jpeg_shape = tf.image.extract_jpeg_shape(image_bytes)
    bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
    image = _distorted_bounding_box_crop(
        image_bytes,
        jpeg_shape=jpeg_shape,
        bbox=bbox,
        min_object_covered=0.1,
        aspect_ratio_range=(3 / 4, 4 / 3),
        area_range=(0.08, 1.0),
        max_attempts=10)
    if tf.reduce_all(tf.equal(jpeg_shape, tf.shape(image))):
        # If the random crop failed fall back to center crop.
        image = _decode_and_center_crop(image_bytes, jpeg_shape)
    return image


def _decode_and_center_crop(
        image_bytes: tf.Tensor,
        jpeg_shape: Optional[tf.Tensor] = None,
) -> tf.Tensor:
    """Crops to center of image with padding then scales."""
    if jpeg_shape is None:
        jpeg_shape = tf.image.extract_jpeg_shape(image_bytes)
    image_height = jpeg_shape[0]
    image_width = jpeg_shape[1]

    # padded_center_crop_size = tf.cast(
    #     ((cropSize / (image_height)) *
    #      tf.cast(tf.minimum(image_height, image_width), tf.float32)), tf.int32)

    offset_height = ((image_height - cropSize) + 1) // 2
    offset_width = ((image_width - cropSize) + 1) // 2
    crop_window = tf.stack([offset_height, offset_width,
                            cropSize, cropSize])
    image = tf.image.decode_and_crop_jpeg(image_bytes, crop_window, channels=3)
    return image





def _distorted_bounding_box_crop(
        image_bytes: tf.Tensor,
        *,
        jpeg_shape: tf.Tensor,
        bbox: tf.Tensor,
        min_object_covered: float,
        aspect_ratio_range: Tuple[float, float],
        area_range: Tuple[float, float],
        max_attempts: int,
) -> tf.Tensor:
    """Generates cropped_image using one of the bboxes randomly distorted."""
    bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(
        jpeg_shape,
        bounding_boxes=bbox,
        min_object_covered=min_object_covered,
        aspect_ratio_range=aspect_ratio_range,
        area_range=area_range,
        max_attempts=max_attempts,
        use_image_if_no_bounding_boxes=True)

    # Crop the image to the specified bounding box.
    offset_y, offset_x, _ = tf.unstack(bbox_begin)
    target_height, target_width, _ = tf.unstack(bbox_size)
    crop_window = tf.stack([offset_y, offset_x, target_height, target_width])
    image = tf.image.decode_and_crop_jpeg(image_bytes, crop_window, channels=3)
    return image

#jax functions! assume input is in range 0,255; although they'll be floats, and possibly f16 if that is set

def auto_contrast(image, key, policy):
    original_image = image

    low = jnp.min(image)
    high = jnp.max(image)
    scale = 255.0 / (high - low)
    offset = -low * scale

    image = image * scale + offset
    result = jnp.clip(image, 0.0, 255.0)

    # don't process NaN channels
    result = jnp.where(jnp.isnan(result), original_image, result)

    return result



def equalize(image, key, policy):
    def equalize_channel(image):
        """equalize_channel performs histogram equalization on a single channel.

        Args:
            image: int Tensor with pixels in range [0, 255], RGB format,
                with channels last
            channel_index: channel to equalize
        """
        nbins = image.shape[0]
        # image = image[..., channel_index]
        # Compute the histogram of the image channel.
        histogram, bin_edges = jnp.histogram(image, range=[0, 255], bins=nbins)

        # For the purposes of computing the step, filter out the nonzeros.
        # Zeroes are replaced by a big number while calculating min to keep shape
        # constant across input sizes for compatibility with vectorized_map

        big_number = 1410065408
        histogram_without_zeroes = jnp.where(jnp.equal(histogram, 0), big_number, histogram)

        step = (jnp.sum(histogram) - jnp.min(histogram_without_zeroes)) // 255
        def build_mapping(histogram, step):
            # Compute the cumulative sum, shifting by step // 2
            # and then normalization by step.
            lookup_table = (jnp.cumsum(histogram) + (step // 2)) // step
            # Shift lookup_table, prepending with 0.
            lookup_table = jnp.concatenate([jnp.array([0]), lookup_table[:-1]])
            # Clip the counts to be in range.  This is done
            # in the C code for image.point.
            return jax.lax.convert_element_type(jnp.clip(lookup_table, 0, 255), jnp.int32)

        # If step is zero, return the original image.  Otherwise, build
        # lookup table from the full histogram and step and then index from it.
        result = jnp.where(
            jnp.equal(step, 0),
            image,
            image[build_mapping(histogram, step)],
        )

        return result


    image = jax.lax.convert_element_type(image, jnp.int32)
    vmapped_equalize_channel = jax.vmap(equalize_channel,
                                        in_axes=2, out_axes=2)
    image = vmapped_equalize_channel(image)

    image = policy.cast_to_compute(image)

    return image

magnitude = 0.5
magnitude_stddev = 0.15

def solarize(image, key, policy):

    maxAddition = 110
    mean = magnitude*maxAddition
    stdDev = magnitude_stddev*maxAddition
    addKey, threshKey = jax.random.split(key)
    addition = jnp.clip(jax.random.normal(addKey) * stdDev + mean, 0,maxAddition)
    mean = 255-(magnitude*255)
    stdDev = magnitude_stddev*255
    threshold = jnp.clip(jax.random.normal(threshKey) * stdDev + mean, 0, 255)

    result = image + addition
    result = jnp.clip(result, 0, 255)
    result = jnp.where(result < threshold, result, 255 - result)

    return result

def solarize_pix(image,key,policy):
    from dm_pix import solarize
    mean = 255 - (magnitude * 255)
    stdDev = magnitude_stddev * 255
    threshold = jnp.clip(jax.random.normal(key) * stdDev + mean, 0, 255)
    image = solarize(
        threshold=threshold,
        image=image)
    return image

def color_degen(image, key, policy):
    factor = jnp.clip(jax.random.normal(key) * magnitude_stddev + magnitude, 0, 1)

    def rgbToGreyscale(image):
        rgb_weights = jnp.array([0.2989, 0.5870, 0.1140])
        greyImg = jnp.tensordot(image, rgb_weights, [-1, -1])
        greyImg = jnp.expand_dims(greyImg, -1) #Probably to add a channel dim
        return greyImg

    def greyscaleToRgb(image):
        rgb = jnp.tile(image, (1,1,3))
        return rgb

    degenImage = greyscaleToRgb(rgbToGreyscale(image))

    difference = degenImage - image
    scaled = factor * difference
    temp = image + scaled
    return jnp.clip(temp, 0.0, 255.0)

def random_contrast(image, key, policy):
    from dm_pix import random_contrast
    low = 1- magnitude
    hi = 1+magnitude
    image = random_contrast(key,image,low,hi)
    return image

# broken dm_pix function
# def random_brightness(image,key,policy):
#     from dm_pix import random_brightness
#     return random_brightness(key,image,magnitude)

def shear(image,key,policy):
    from imax.transforms import apply_transform,shear
    key1,key2,key3,key4=jax.random.split(key,4)
    mean = magnitude
    stdDev = magnitude_stddev
    x = jnp.clip(jax.random.normal(key1) * stdDev + mean, 0, 1)
    y = jnp.clip(jax.random.normal(key2) * stdDev + mean, 0, 1)
    x = jnp.where(jax.random.uniform(key3) < 0.5, -x, x)
    y = jnp.where(jax.random.uniform(key4) < 0.5, -y, y)

    image = apply_transform(image, shear(horizontal=x,
                                                               vertical=y))

    return image



def translate(image,key,policy):
    from imax.transforms import translate, apply_transform
    key1, key2 = jax.random.split(key)

    x = jax.random.uniform(key1,minval=-magnitude*.3,maxval=magnitude*.3)
    y = jax.random.uniform(key2,minval=-magnitude*.1,maxval=magnitude*.1)
    image = apply_transform(image, translate(horizontal=x,
                                                               vertical=y))
    return image

def posterize(image, key, policy):
    from imax.color_transforms import posterize
    level = jax.random.uniform(key) * magnitude
    bits = 5 - jnp.min(jnp.array([4, (level * 4).astype('uint8')]))
    image = posterize(image, bits)
    return image

def brightness(image,key,policy):
    from imax.color_transforms import brightness
    factor = jax.random.uniform(key) * magnitude
    return brightness(image, factor)

def sharpness(image, key, policy):
    from imax.color_transforms import sharpness
    factor = jax.random.uniform(key) * magnitude
    return sharpness(image, factor)

def rotate(image, key, policy):
    from imax.transforms import rotate, apply_transform
    factor = jax.random.uniform(key, minval=-magnitude, maxval=magnitude) * jnp.pi/2
    image = apply_transform(image, rotate(factor))
    return image

def flipX(image, key, policy):
    from imax.transforms import flip,apply_transform
    return apply_transform(image, flip(True, False))

def cutout(image,key, policy):
    from imax.color_transforms import get_random_cutout_mask, cutout
    cutout_mask = get_random_cutout_mask(
        key,
        image.shape,
        image.shape[0]//3)
    return cutout(image, cutout_mask)

randomLayers = [
    flipX,
    translate,
    color_degen,
    auto_contrast,
    equalize,
    random_contrast,
    brightness,
    cutout,
    rotate,
]

def randomAugmentJax(images, rng, policy, augmentations_per_image = 3, rate = 10/11,
                     thruLayer = -1, onlyLayer = -1):
    # For each image, we're going to want a different rng, so we need to split
    keys = jax.random.split(rng, images.shape[0])

    if thruLayer == -1:
        if onlyLayer == -1:
            myLayers = randomLayers
        else:
            myLayers = randomLayers[onlyLayer]
    else:
        myLayers = randomLayers[:thruLayer+1]

    def apply_rand_layer(image, layer_select_key, layer_key):
        randIndex = jax.random.randint(layer_select_key, (), 0, len(myLayers)-1)
        for i in range(len(myLayers)):
            image = jnp.where(randIndex == i, myLayers[i](image, layer_key, policy), image)
        # todo: may want to convert the above to jax.lax.scan over indices so jax.jit doesn't unroll it; faster compilation
        #  however, first attempt did not go well; it may not be possible due to the branching (jnp.where)
        return image

    # Now, we need to define a function we'll vmap to our images
    def augment(image, loop_key):

        #within this function, we just copy the rand_augment apply
        original_image = image

        for _ in range(augmentations_per_image):
            # get a skip uniform random number
            loop_key, skip_key, layer_select_key, layer_key = jax.random.split(loop_key,4)
            skip_augment = jax.random.uniform(skip_key)
            #skip based on rng above (identity)
            #else, apply a random layer to the input
            image = jnp.where(skip_augment > rate,
                              original_image,
                              apply_rand_layer(image, layer_select_key, layer_key))
        # todo: same as above (jax.lax.scan)
        return image

    return jax.vmap(augment)(images, keys)
