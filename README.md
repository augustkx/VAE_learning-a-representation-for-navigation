# AVE_learning-a-representation-for-navigation
The training data is the seqence of image exracting from a house tour video in https://www.youtube.com/watch?v=jdNiWiXiJQ4\&t=200s

Here are the procedures for prearing training images:

1. cut the viseo into shorter pieces:
ffmpeg -ss 0:0:37 -i input.mp4 -t 0:3:20  -c copy out.mp4 

2. extract 1000 images from the cut video.
  ffmpeg -i out.mp4 -vf fps=1000/3.33*60 %d.png,


After the training data is ready. Run the training_model_vae_deconv _4d_latent.py,  you will get the visulaization of learned manifold and also a 'h5' file that stores the trained VAE model parameter information. This file will be used in the path planning stage.

Note, that you should change the address in the code, where the training data is stored in your system. 

After the VAE model is learned, it can be used as a representation map for navigation.
Run the navigation_main.py file.

First, in  line 
