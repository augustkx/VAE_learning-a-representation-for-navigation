# Learning a Representation Map for Robot Navigation using Deep Vatiational Autoencoder

### Training VAE


The training data is the set of images exracting from a house tour video in https://www.youtube.com/watch?v=jdNiWiXiJQ4\&t=200s

Here are the procedures for prearing training images:

1. cut the viseo into shorter pieces:
```
ffmpeg -ss 0:0:37 -i input.mp4 -t 0:3:20  -c copy out.mp4 
```

2. extract 1000 images from the cut video:
  ```
  ffmpeg -i out.mp4 -vf fps=1000/3.33*60 %d.png
  ```

After the training data is ready. Run the training_model_vae_deconv _4d_latent.py,  you will get the visulaization of learned manifold and also a 'h5' file that stores the trained VAE model parameter information. This file will be used in the path planning stage. Note, that in the code you should change the address, where the training data is stored in your system. 


### Path Planning

After the VAE model is learned, it can be used as a representation map for navigation.
Run the navigation_main.py file. In line 30 and 31, you can change the starting image and ending image. At the bottom of the file, there are some functions about path planning method, whose effects can be in four categories:

1.Initail path in Euclidean space, visualise path with generated images

2.Gradient descent method on the path sequence to update it to geodesic, and visualize the path with generated images 

3.Visualize the route with real frames according to generated images

4.Select a ground truth route by humans as the basis for evaluation

**The experiments described in the thesis are conducted as following:**

First, we try to produce a straigh line in the 4 D latent Euclidean space, connecting the staring point(in the latent space) and ending point(in the latent space), and then project it to the manifold to get the path. Then, a corresponding route is generated. To do this, uncomment: ```navi.visualise_route('passway_straight.png',points,shrink,batch_size)```in part 1.

Second, we can construct route with real frames with the nearest neighbour metric. This is done by mapping each reconstructed image in the sequence with the images in the training set. To do this, umcomment: 
```navi.visualise_route('passway_straight.png',points,shrink,batch_size)``` in part 1;
```navi.route_reality('straight_line',shrink,points)``` in part 2.

Third, we try to update the path generated in Euclidean space to geodesic by Gradient Descent method. To do this, umcomment: 
```navi.visualise_route('passway_dg.png',points,shrink,batch_size,num_p)``` in part 1; and ```POINTS= navi.gd_points(points,num_p)``` in part 2.


Forth, we can construct route with real frames with the nearest neighbour metric. To do this, umcomment: 
```navi.visualise_route('passway_straight.png',points,shrink,batch_size)```
```POINTS= navi.gd_points(points,num_p)``` 
```navi.visualise_route('passway_dg.png',points,shrink,batch_size,num_p)```in part 1;
and ```POINTS=navi.route_reality('Gredient_descent',shrink,points)``` in part 2.


Fifth, the path produced through the learned latent space manifold is reasonable and short, but not continously enough for robot navigate with computer vision techniques. A ground truth path / route is generated by human selection, and is used as a basis for evaluating the experiment route.
To do this: uncomment all the functions in part 3. 

Lastly, to evaluate the results: uncomment all the functions in part 4.




# Reference
https://github.com/keras-team/keras/blob/master/examples/variational_autoencoder_deconv.py




