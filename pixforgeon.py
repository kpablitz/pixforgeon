#!/usr/bin/env python3

import os
import argparse
import tensorflow as tf
from utils import *
from myvgg import * 

WEIGHT_PATH= 'pretrained_model/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5'
STYLE_WEIGHT=1e-2
CONTENT_WEIGHT=1e4
CONTENT_LAYERS = ['block5_conv2'] 

STYLE_LAYERS = ['block1_conv1',
                'block2_conv1',
                'block3_conv1', 
                'block4_conv1', 
                'block5_conv1']


num_content_layers = len(CONTENT_LAYERS)
num_style_layers = len(STYLE_LAYERS)
TOTAL_VARIATION_WEIGHT=30

def validate_paths(paths):
    for path in paths:
        if not os.path.exists(path):
            raise ValueError(f"Error: Path does not exist: {path}")

def process_arguments():
    # Creating an ArgumentParser object
    parser = argparse.ArgumentParser(description='Image Processing Script')

    # Adding arguments
    parser.add_argument('--content-image', type=str, help='Path to the content image file, including the image name. Example: /path/to/image/your_content_image.jpg')
    parser.add_argument('--style-image', type=str, help='Path to the content image file, including the image name. Example: /path/to/image/your_style_image.jpg')
    # Adding an optional argument
    parser.add_argument('--epochs', type=int, default=400, help='Number of echoes. Default: 400')
    parser.add_argument('--output-filename', type=str, default='stylized_image.jpg',
                        help='Specify the output filename for the generated image. Default: stylized_image.jpg')
    parser.add_argument('--learning-rate', type=float, default=0.01, help='Set the learning rate. Default: 0.01')

    # Parsing the arguments
    args = parser.parse_args()

    # Accessing the arguments
    content_image_path = args.content_image
    style_image_path = args.style_image
    epochs = args.epochs
    learning_rate = args.learning_rate
    if not args.output_filename.endswith('.jpg'):
        args.output_filename += '.jpg'
    output_path = os.path.join("output_images", args.output_filename)
    # Create the output folder if it doesn't exist
    os.makedirs("output_images", exist_ok=True)
    output_path = f"{output_path}"

    # Validate paths
    validate_paths([content_image_path, style_image_path])

    # Validate epochs
    if not isinstance(epochs, int):
        raise ValueError(f"Error: 'epochs' must be an integer")

    return content_image_path, style_image_path, epochs, learning_rate, output_path

def gram_matrix(input_tensor):
    # Reshape the input tensor (batch, h,w,c ) to (batch_size, height*width, channels)
    reshaped_input = tf.reshape(input_tensor, (tf.shape(input_tensor)[0], -1, tf.shape(input_tensor)[-1]))
    # Transpose the reshaped tensor to get (batch_size, channels, height*width)
    transposed_input = tf.transpose(reshaped_input, perm=[0, 2, 1])
    # Compute the Gram matrix using tf.matmul
    result = tf.matmul(transposed_input, reshaped_input)
    # Normalize by the number of locations
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
    return result / num_locations

def vgg_layers(layer_names):
  """ Creates a VGG model that returns a list of intermediate output values."""
  # Load our model. Load pretrained VGG, trained on ImageNet data
  vgg = MyVGG(include_top=False, weights=WEIGHT_PATH)
  vgg.trainable = False
  outputs = [vgg.model.get_layer(name).output for name in layer_names]
  model = tf.keras.Model([vgg.inputs], outputs)
  return model


def style_content_loss(outputs,style_features,content_features):
    style_outputs = outputs['style']
    content_outputs = outputs['content']
    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name]-style_features[name])**2) 
                           for name in style_outputs.keys()])
    style_loss *= STYLE_WEIGHT / num_style_layers

    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]-content_features[name])**2) 
                             for name in content_outputs.keys()])
    content_loss *= CONTENT_WEIGHT / num_content_layers
    loss = style_loss + content_loss
    return loss


class ArtisticFeatureExtractor(tf.keras.models.Model):
  def __init__(self, style_layers, content_layers):
    super(ArtisticFeatureExtractor, self).__init__()
    self.vgg = vgg_layers(style_layers + content_layers)
    self.style_layers = style_layers
    self.content_layers = content_layers
    self.num_style_layers = len(style_layers)
    self.vgg.trainable = False

  def call(self, inputs):
    "Expects float input in [0,1]"
    inputs = inputs*255.0
    # VGG19 is trained using the Caffe framework
    preprocessed_input = preprocess_input_caffe(inputs)
    outputs = self.vgg(preprocessed_input)
    style_outputs, content_outputs = (outputs[:self.num_style_layers],
                                      outputs[self.num_style_layers:])

    style_outputs = [gram_matrix(style_output)
                    for style_output in style_outputs]

    content_dict = {content_name: value
                    for content_name, value
                    in zip(self.content_layers, content_outputs)}

    style_dict = {style_name: value
                  for style_name, value
                  in zip(self.style_layers, style_outputs)}

    return {'content': content_dict, 'style': style_dict}

def main():
    content_image_path, style_image_path, epochs, learning_rate, output_path = process_arguments()
    content_image = load_img(content_image_path)
    style_image = load_img(style_image_path)
    feature_extractor = ArtisticFeatureExtractor(STYLE_LAYERS, CONTENT_LAYERS)

    style_features = feature_extractor(style_image)['style']
    content_features = feature_extractor(content_image)['content']


    @tf.function()
    def train_step(image):
      with tf.GradientTape() as tape:
        outputs = feature_extractor(image)
        loss = style_content_loss(outputs,style_features,content_features)
        loss += TOTAL_VARIATION_WEIGHT*tf.image.total_variation(image)

      grad = tape.gradient(loss, image)
      opt.apply_gradients([(grad, image)])
      image.assign(clip_0_1(image))


    image = tf.Variable(content_image)
    tf.image.total_variation(image).numpy()
    opt = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate, beta_1=0.99, epsilon=1e-1)

    for i in range(epochs):
        train_step(image)
    image = tensor_to_image(image)
    #plt.imshow(image)
    image.save(f"{output_path}")

    pass

if __name__ == "__main__":
    main()
