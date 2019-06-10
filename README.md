Generating Cats and Flowers with GANs
============================================
A few months ago I saw a [blog post](https://ajolicoeur.wordpress.com/cats/) 
about generating cat faces with different kinds of generative adversarial networks. 
I never trained a WGAN before, and I had some free time on my hands 
so I decided rewrite the code and try it out myself.
I skipped a lot of the parameter tuning and just used the hyper-parameters described in the blog post. 


### Requirements
- python 3.6
- opencv-python 
- numpy
- **pytorch 1.1.0 is a must**

### Experiemnts
I used DCGAN and WGAN (with gradient penalty). 
Additionally to the [cat dataset](https://www.kaggle.com/crawford/cat-dataset),
I also experimented with generating flowers using the [102 category flower dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/).
In both cases I downsampled the images to 64x64, 
and since the datasets contain roughly the same number of images, I used the same hyper-parameters.

### Results
For both datasets and networks the training was stable,
and I ended up with more-or-less good looking, believable cats/flowers.
I trained the DCGAN models for 280 epoch, and the WGAN-GP models for about 700.

![alt-text-1](imgs/dcgan_cats_ep280_sample.png "DCGAN-cats") ![alt-text-2](imgs/wgan_cats_ep700_sample.png "WGAN_GP-cats")

<img src="imgs/dcgan_cats_ep280_sample.png" width="300"> <img src="imgs/wgan_cats_ep700_sample.png" width="300">
