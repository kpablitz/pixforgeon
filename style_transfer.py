import tensorflow as tf
from utils import *
from myvgg import * 

weight_path= 'pretrained_model/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5'
style_weight=1e-2
content_weight=1e4
# Define layers 
content_layers = ['block5_conv2'] 

style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1', 
                'block4_conv1', 
                'block5_conv1']

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)


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
  vgg = MyVGG(include_top=False, weights=weight_path)
  vgg.trainable = False
  outputs = [vgg.model.get_layer(name).output for name in layer_names]
  model = tf.keras.Model([vgg.inputs], outputs)
  return model


# Load content and style images
content_image = load_img('images/maru.png')
style_image = load_img('images/okeffe.jpg')


class StyleContentModel(tf.keras.models.Model):
  def __init__(self, style_layers, content_layers):
    super(StyleContentModel, self).__init__()
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

extractor = StyleContentModel(style_layers, content_layers)

style_targets = extractor(style_image)['style']
content_targets = extractor(content_image)['content']
image = tf.Variable(content_image)

def style_content_loss(outputs):
    style_outputs = outputs['style']
    content_outputs = outputs['content']
    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name]-style_targets[name])**2) 
                           for name in style_outputs.keys()])
    style_loss *= style_weight / num_style_layers

    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]-content_targets[name])**2) 
                             for name in content_outputs.keys()])
    content_loss *= content_weight / num_content_layers
    loss = style_loss + content_loss
    return loss

total_variation_weight=30
tf.image.total_variation(image).numpy()


@tf.function()
def train_step(image):
  with tf.GradientTape() as tape:
    outputs = extractor(image)
    loss = style_content_loss(outputs)
    loss += total_variation_weight*tf.image.total_variation(image)

  grad = tape.gradient(loss, image)
  opt.apply_gradients([(grad, image)])
  image.assign(clip_0_1(image))

opt = tf.keras.optimizers.legacy.Adam(learning_rate=0.01, beta_1=0.99, epsilon=1e-1)
image = tf.Variable(content_image)

epochs = 500
for i in range(epochs):
    train_step(image)
image = tensor_to_image(image)
#plt.imshow(image)
image.save(f"output/maru1.jpg")