import numpy as np
import tensorflow as tf
from PIL import Image
import PIL.Image
import matplotlib.pyplot as plt

def load_img(path_to_img):
    # Set the maximum dimension for resizing
    max_dim = 512
    # Read the image file
    img = tf.io.read_file(path_to_img)
    # Decode the image to a 3-channel tensor
    img = tf.image.decode_image(img, channels=3)
    # Convert the pixel values to float32 between 0 and 1
    img = tf.image.convert_image_dtype(img, tf.float32)
    # Get the shape of the image and cast to float32
    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    # Find the longest dimension and calculate the scale factor
    long_dim = max(shape)
    scale = max_dim / long_dim
    # Calculate the new shape after resizing
    new_shape = tf.cast(shape * scale, tf.int32)
    # Resize the image with the calculated new shape
    img = tf.image.resize(img, new_shape)
    # Add an extra dimension to the tensor to represent batch size
    img = img[tf.newaxis, :]
    # Return the processed image tensor
    return img


def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)


# Caffe data format preprocess step
def preprocess_input_caffe(x, data_format='channels_last'):
    """
    Preprocesses an input image tensor to match the Caffe framework's expectations.

    Parameters:
    - x: Input image tensor.
    - data_format: Data format, either 'channels_last' or 'channels_first'.

    Returns:
    - Preprocessed image tensor.
    """
    # Ensure 'channels_last' data format if not specified
    if data_format == 'channels_last':
        # Reverse the order of color channels (RGB to BGR)
        x = x[..., ::-1]
    else:
        # Reverse the order of color channels (RGB to BGR)
        x = x[:, ::-1, :]

    # Convert RGB to BGR and zero-center each color channel
    mean = [103.939, 116.779, 123.68]
    x -= mean
    return x

def imshow(image, title=None):
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis=0)

    plt.imshow(image)
    if title:
        plt.title(title)   

def clip_0_1(image):
    """
    Clips the pixel values of an image tensor to the range [0.0, 1.0].

    Parameters:
    - image: Input image tensor.

    Returns:
    - Tensor with pixel values clipped to the range [0.0, 1.0].
    """
    # Clip the pixel values of the image tensor to the range [0.0, 1.0]
    clipped_image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)
    return clipped_image