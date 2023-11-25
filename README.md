#  About                   <img src="images/logo.png" alt="Logo" width="90" height="55" style="margin: -1% 1%"> 
<img align="right" width="300" height="auto" src="images/animation.gif">

**pixforgeon** is my experimental implementation of **Neural Style Transfer (NST)** using `TensorFlow`. `myvgg.py` was first developed following the guidelines of the original [**VGG19**](https://arxiv.org/pdf/1409.1556.pdf) implementation. Subsequently, the **NST** technique was utilized, blending the style of one image into the content of another, resulting in visually mesmerizing compositions.

## Table of Contents
  - [Synopsis](#synopsis)
  - [Personal Remarks](#personal-remarks)
  - [Repo Structure](#repo-structure)
  - [Setup](setup)
    - [TensorFlow Intricacies](#tensorflow-intricacies)
  - [Roll it!](#roll-it)
  - [M1 CPU vs GPU](#m1-cpu-vs-gpu)
  - [Logo](#logo)
  - [Acknowlegments](#Acknowlegments)
  - [License](#license)

## Synopsis
**NST** is an innovative technique that performs the following steps:

1. **Input Images:** Select a content image and a style reference image.

2. **Feature Extraction:** Employ a pre-trained neural network, **VGG19** in this case, to extract features from both the content and style images. This process involves selecting specific layers that capture meaningful information. Typically, the initial layers' activations capture basic or low-level features, whereas the activations in the final layers represent more complex or high-level features.

3. **Loss Calculation:** Calculate the loss function to measure the disparity between the generated/stylized image and both the content and style of the reference image. 

4. **Optimization:** Apply an optimization algorithm to minimize the overall loss by adjusting the pixel values of the generated image.

5. **Output Image:** The final output is a stylized image that combines the content of the chosen image with the artistic style of the reference image.

## Personal Remarks
My main sources of applying `step 3` was an exercise in a course offered by DeepLearning.AI in the Coursera platform instructed by Andrew Ng and  [this](https://www.tensorflow.org/tutorials/generative/style_transfer) tutorial by TensorFlow. 

Since that and because I noticed an inconsistency between the 2 implementations which is that in the latter, the `gram` matrix is not applied to the content image, I checked few sources to understand the reason. It turns out that while the `gram` matrix is traditionally associated with style loss to capture feature correlations for style, some implementations experiment with using it in conjunction with the content image to achieve specific artistic effects. This is not a standard approach but can be explored for creative purposes.

By applying the second technique and experimenting with different images on actual faces, I noticed that based on image background, the style sometimes can take over quite fast (approx. 400 epochs). Sometimes satisfying results were gernerated, while other times for epochs > 2000 and a learning rate of 0.01, the style could distort the image, leading to either fascinating or funny outcomes. Initial configurations for `style_weight`, `content_weight`, and `total_variation_weight` were `1e-2`, `1e4`, `30` respectively. 

As a next step, I experimented with the above settings giving more `weight` to style and content to either emphasize style features and converge to an artistic result faster (fewer epochs) or get ride of content distortions by preserving content.

In further exploration, I drew inspiration from the insightful experiments conducted by Aleksa Gordić and played with `total variation weight`. His [work](https://github.com/gordicaleksa/pytorch-neural-style-transfer/tree/master) provides valuable insights into the effects of manipulating style, content and total variation weights. 
 

## Repo Structure
```
pixforgeon/
├── README.md
├── images/
│   ├── logo.png
│   ├── content_images/
│   ├── style_images/
├── myvgg.py
├── output_images/
├── pixforgeon.py
├── pretrained_model/
│   └──vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5 
├── requirements.txt
└── utils.py
```

## Setup

To set up this project locally, follow these steps:

```bash
# Step 1: Clone the repository
git clone https://github.com/kpablitz/pixforgeon.git
cd pixforgeon

# Step 2: Create a virtual environment
python -m venv .venv

# Step 3: Activate the virtual environment on Unix or MacOS
source .venv/bin/activate

# Step 4: Install dependencies
pip install -r requirements.txt
```

### Tensorflow Intricacies

For optimal performance, it is advisable to use `tensorflow` with a `gpu` when available. Otherwise, training models with a high number of `epochs` may take an extended amount of time.

I tested this with version `2.14.0`, and starting from version `2.x.x`, the `gpu` version for `NVIDIA GPUs` using `CUDA` is integrated into the package.

If your system does not have an `NVIDIA GPU`, you may receive warning messages that can be safely ignored, and the package will utilize the `cpu`.

In my case, I used the package along with the `tensorflow-metal` package for `Macs` with `M1` chips.

For those interested in utilizing `AMD GPUs` with `tensorflow`, you can explore ROCm support [here](https://rocm.docs.amd.com/en/latest/how_to/tensorflow_install/tensorflow_install.html).




## Roll it!

Enhance your images with **NST** using the pixforgeon.py script. Explore the available options by typing `--help` to customize and optimize the application of NST to your pictures.



```python
options:
  -h, --help            show this help message and exit
  --content-image CONTENT_IMAGE, -c CONTENT_IMAGE
                        Path to the content image file, including the image name. Example: /path/to/image/your_content_image.jpg
  --style-image STYLE_IMAGE, -s STYLE_IMAGE
                        Path to the content image file, including the image name. Example: /path/to/image/your_style_image.jpg
  --content-weight CONTENT_WEIGHT, -cw CONTENT_WEIGHT
                        Set the weight for content loss. Default: 1e5
  --style-weight STYLE_WEIGHT, -sw STYLE_WEIGHT
                        Set the weight for style loss. Default: 1e1
  --epochs EPOCHS, -e EPOCHS
                        Number of echoes. Default: 400
  --output-filename OUTPUT_FILENAME
                        Specify the output filename for the generated image. Default: stylized_image.jpg
  --learning-rate LEARNING_RATE, -lr LEARNING_RATE
                        Set the learning rate. Default: 0.01
  --total-variation-weight TOTAL_VARIATION_WEIGHT, -tw TOTAL_VARIATION_WEIGHT
                        Set the weight for total variation loss. Default: 1e1
```

Fields that have Default values are optional.

Give execute rights to `chmod +x pixforgeon.py` or use `python pixforgeon.py`  

Run: 

```bash
./pixforgeon.py --content-image /path/to/image/your_content_image.jpg --style-image /path/to/image/your_style_image.jpg
```

I have tested it with `jpg` `jpeg` `png` formats.

You can configure the name of the generated image with the option `--output-filename OUTPUT_FILENAME`. The generates image is stored under `./output_images/` folder which is created in case it is missing.

![NST_Output_Plot](images/NST_outputs_plot.jpg)


45.20 metal, 436.42 seconds

# M1 CPU vs GPU

Performance test conducted with `--epochs 500`, comparing the `Apple M1 chip's` `cpu` and `gpu`. We can see that the performance boost by using the GPU is around `9,6` times faster.

| Component | Time Taken |
|-----------|------------|
| M1 CPU    | 433.16 seconds |
| M1 GPU    | 45.20 seconds  |

Note: During the implementation phase, a warning message was observed, indicating that the performance of `tf.keras.optimizers.Adam` was slower on `M1/M2 Macs` than the legacy `tf.keras.optimizers.legacy.Adam` and thus the latter is used.


## Logo
Logo was generated on [here](https://logo.com/).

## Acknowlegments
- [DeepLearning](https://www.deeplearning.ai/courses/deep-learning-specialization/) Specialization by Andrew Ng 

- VGG [paper](https://arxiv.org/pdf/1409.1556.pdf)

- Tensorflow's [tutorial on NST](https://www.tensorflow.org/tutorials/generative/style_transfer).

- Pytorch [NST by Aleksa Gordić ](https://github.com/gordicaleksa/pytorch-neural-style-transfer) for insights and showcasing on the effects on weight parameters

- Pytorch [NST by Nazia Nafis](https://github.com/nazianafis/Neural-Style-Transfer/tree/main) 

- Keras [implementation](https://hub.packtpub.com/neural-style-transfer-creating-artificial-art-with-deep-learning-and-transfer-learning/)


## License 

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)