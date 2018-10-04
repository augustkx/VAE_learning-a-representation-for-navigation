from navigation_function import encoder,generator
import navigation_function as navi

from keras.callbacks import EarlyStopping, ModelCheckpoint
from PIL import Image
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from keras.layers import Input, Dense, Lambda, Flatten, Reshape, Layer,Dropout
from keras.layers import Conv2D, Conv2DTranspose,MaxPooling2D
from keras.models import Model
from keras import backend as K
from keras import metrics

from keras.datasets import mnist
from keras.models import model_from_json


shrink=60# input image dimensions
img_rows, img_cols, img_chns = shrink,shrink, 3
batch_size = 20
if K.image_data_format() == 'channels_first':
    original_img_size = (img_chns, img_rows, img_cols)
else:
    original_img_size = (img_rows, img_cols, img_chns)
latent_dim = 4
# navi.vis()#visualization of the manifold.
#==============================================================================================
start=1#the number represents the corresponding image in the training data i.e. the image are stored in the training collection numbered from 1 to 1000.
destination=300# the number represents the corresponoding image in training data.
image_s=[]#starting image 

image = Image.open('/home/petered/projects/Auto-encoder/pictures/'+ str(start) +'.jpg')#change the path to your file address where image data are stored.
# image = Image.open('/home/kaixin/vae-map/vae_map/pictures_new/' + str(start) + '.jpg')

image = image.resize((shrink, shrink))
image=np.asarray(image)
image=image.tolist()
image_s.append(image)
image_s=np.asarray(image_s)
image_s= np.tile(image_s, ( batch_size,1,1,1))

image_d=[]#destination images
image = Image.open('/home/petered/projects/Auto-encoder/pictures/'+ str(destination) +'.jpg')#change the path to your file address where image data are stored.

# image = Image.open('/home/kaixin/vae-map/vae_map/pictures_new/' + str(destination) + '.jpg')
image = image.resize((shrink, shrink))
image=np.asarray(image)
image=image.tolist()
image_d.append(image)
image_d=np.asarray(image_d)
image_d= np.tile(image_d, ( batch_size,1,1,1))



image_d = image_d.astype('float32') / 255.
image_s = image_s.astype('float32') / 255.
image_d = image_d.reshape((image_d.shape[0],) + original_img_size)
image_s = image_s.reshape((image_d.shape[0],) + original_img_size)
#
z_s= encoder.predict(image_s,batch_size=batch_size)[0]
z_d= encoder.predict(image_d,batch_size=batch_size)[0]

distance=z_d-z_s
print(z_s,z_d)


num_p=50
# points that connects the staring image representation and destination image representation in the latent space.
points = (z_d - z_s) * np.linspace(0, 1, num_p)[:, None] + z_s # A (num_p, 4) array of points
print(points[1],'checking producing straight line points in latent space')


# =======1.gradient descent method on the path sequence, and visualize the path with generated images ==========
# navi.visualise_route('passway_straight.png',points,shrink,batch_size)#initialize
# gradient descent on the path sequence
#should first run the code navi.visualise_route of straight line, for the sake of producing straight line sequence.
points= navi.gd_points(points,num_p)
navi.visualise_route('passway_dg.png',points,shrink,batch_size,num_p)
# ============================================================

# ==== 2. visualize the route with real frames according to generated images======================
POINTS=navi.route_reality('Gredient_descent',shrink,points)
# navi.route_reality('straight_line',shrink,points)
#
#=====3. select a route by hand and calculate the loss.===================================
# hard coded. visualisation included.
digits=navi.route_manually(num_p,shrink,file='manually selected')
# feed them through encoder
digits_encoded= encoder.predict(digits, batch_size=batch_size)

# feed them through the decoder
navi.visualise_route('reconstructed_from_manually',digits_encoded,shrink,batch_size,num_p)
navi.metric(digits_encoded,shrink,batch_size)

# ============4.evaluate and compare=========================

navi.evaluate(points,digits_encoded,POINTS,digits,shrink,batch_size)




