

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

# input image dimensions
shrink=60
img_rows, img_cols, img_chns = shrink,shrink, 3

filters =32# number of convolutional filters to use

num_conv = 3
stri=1 #stride value of last con layer;set it none.
stri2=1

batch_size = 20
if K.image_data_format() == 'channels_first':
    original_img_size = (img_chns, img_rows, img_cols)
else:
    original_img_size = (img_rows, img_cols, img_chns)
latent_dim = 4
intermediate_dim = 256
epsilon_std = 1.0
# epochs = 500

x = Input(shape=original_img_size)
conv_1 = Conv2D(img_chns,
                kernel_size=(2, 2),
                kernel_initializer='random_uniform',
                padding='same', activation='relu')(x)
conv_2 = Conv2D(filters,
                kernel_size=(2, 2),
                kernel_initializer='random_uniform',
                padding='same', activation='relu',
                strides=(2, 2))(conv_1)
# max_pool1 = MaxPooling2D(pool_size=(2, 2),strides=None, padding='valid', data_format=None)(conv_2)
conv_3 = Conv2D(filters,
                kernel_size=num_conv,
                kernel_initializer='random_uniform',
                padding='same', activation='relu',
                strides=(stri2, stri2))(conv_2)
# max_pool2 = MaxPooling2D(pool_size=(2, 2),strides=None, padding='valid', data_format=None)(conv_3)
conv_4 = Conv2D(filters,
                kernel_size=num_conv,
                kernel_initializer='random_uniform',
                padding='same', activation='relu',
                strides=(stri, stri))(conv_3)
# max_pool3 = MaxPooling2D(pool_size=(2, 2),strides=None, padding='valid', data_format=None)(conv_4)

flat = Flatten()(conv_4)
hidden = Dense(intermediate_dim, activation='relu')(flat)
h=Dropout(0.5)(hidden)
z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)


def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim),
                              mean=0., stddev=epsilon_std)
    return z_mean + K.exp(z_log_var) * epsilon

# note that "output_shape" isn't necessary with the TensorFlow backend
# so you could write `Lambda(sampling)([z_mean, z_log_var])`
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

# we instantiate these layers separately so as to reuse them later
decoder_hid = Dense(intermediate_dim,input_shape=(latent_dim,), activation='relu')
# decoder_upsample = Dense(filters2 *( int(shrink/(stri*stri2*2))+0 )* (int(shrink/(stri*stri2*2)+0)), activation='relu')
decoder_upsample = Dense(filters * (shrink/2) * (shrink/2), activation='relu')

if K.image_data_format() == 'channels_first':
    # output_shape = (batch_size, filters2,int( shrink/(stri*stri2*2))+0,int(shrink/(stri*stri2*2))+0)
    output_shape = (batch_size, filters, (shrink/2),(shrink/2))
else:
    # output_shape = (batch_size, int(shrink/(stri*stri2*2))+0, int(shrink/(stri*stri2*2))+0, filters2)
    output_shape = (batch_size, (shrink/2), (shrink/2), filters)

decoder_reshape = Reshape(output_shape[1:])
decoder_deconv_1 = Conv2DTranspose(filters,
                                   kernel_size=num_conv,
                                   kernel_initializer='random_uniform',
                                   padding='same',
                                   strides=(stri, stri),
                                   activation='relu')
decoder_deconv_2 = Conv2DTranspose(filters,
                                   kernel_size=num_conv,
                                   kernel_initializer='random_uniform',
                                   padding='same',
                                   strides=(stri2, stri2),
                                   activation='relu')
if K.image_data_format() == 'channels_first':
    output_shape = (batch_size, filters, 29, 29)
else:
    output_shape = (batch_size, 29, 29, filters)
decoder_deconv_3_upsamp = Conv2DTranspose(filters,
                                          kernel_size=(3, 3),
                                          kernel_initializer='random_uniform',
                                          strides=(2, 2),
                                          padding='valid',
                                          activation='relu')
decoder_mean_squash = Conv2D(img_chns,
                             kernel_size=2,
                             kernel_initializer='random_uniform',
                             padding='valid',
                             activation='sigmoid')

hid_decoded = decoder_hid(z)
up_decoded_=Dropout(0.5)(hid_decoded)
up_decoded = decoder_upsample(up_decoded_)

reshape_decoded = decoder_reshape(up_decoded)
deconv_1_decoded = decoder_deconv_1(reshape_decoded)
deconv_2_decoded = decoder_deconv_2(deconv_1_decoded)
x_decoded_relu = decoder_deconv_3_upsamp(deconv_2_decoded)
x_decoded_mean_squash = decoder_mean_squash(x_decoded_relu)


# Custom loss layer
class CustomVariationalLayer(Layer):
    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(CustomVariationalLayer, self).__init__(**kwargs)

    def vae_loss(self, x, x_decoded_mean_squash):
        x = K.flatten(x)
        x_decoded_mean_squash = K.flatten(x_decoded_mean_squash)
        xent_loss = img_rows * img_cols * metrics.binary_crossentropy(x, x_decoded_mean_squash)
        kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return K.mean(xent_loss + kl_loss)

    def call(self, inputs):
        x = inputs[0]
        x_decoded_mean_squash = inputs[1]
        loss = self.vae_loss(x, x_decoded_mean_squash)
        self.add_loss(loss, inputs=inputs)
        # We don't use this output.
        return x


y = CustomVariationalLayer()([x, x_decoded_mean_squash])
vae = Model(x, y)
vae.compile(optimizer='rmsprop', loss=None)
vae.summary()



vae.load_weights('vae_conv-4d.h5')


# build a model to project inputs on the latent space
encoder = Model(x, z_mean)


# build a digit generator that can sample from the learned distribution
decoder_input = Input(shape=(latent_dim,))
_hid_decoded = decoder_hid(decoder_input)
_up_decoded_=Dropout(0.5)(up_decoded)
_up_decoded = decoder_upsample(_hid_decoded)
_reshape_decoded = decoder_reshape(_up_decoded)
_deconv_1_decoded = decoder_deconv_1(_reshape_decoded)
_deconv_2_decoded = decoder_deconv_2(_deconv_1_decoded)
_x_decoded_relu = decoder_deconv_3_upsamp(_deconv_2_decoded)
_x_decoded_mean_squash = decoder_mean_squash(_x_decoded_relu)
generator = Model(decoder_input, _x_decoded_mean_squash)


# NOTE for variable setting in this script: points always in latent space, 4 d
#       POINTS , digits in 60*60*3 sample space.

def visualise_route(file,points,shrink,batch_size,latent_dim):
    #file: is a tring of the name of output image.
    #visualise from the latent space points to 60*60 reality.
    #fixed 50 images.

    digit_size = shrink
    figure = np.zeros((digit_size*5, digit_size*10,3))
    j=0
    for i, xi in enumerate(points):


        if i in [10,20,30,40,50]:
            j += 1

        z_sample = np.array([[xi]])

        z_sample = np.tile(z_sample, batch_size).reshape(batch_size, latent_dim)
        x_decoded = generator.predict(z_sample, batch_size=batch_size)

        digit = x_decoded[0].reshape(digit_size, digit_size,3)
        if i>39:
            i=i-40
        elif i>29:
            i=i-30
        elif i > 19:
            i=i-20
        elif i>9:
            i=i-10
        # print(i)

        figure[j * digit_size: (j + 1) * digit_size,
               i * digit_size: (i + 1) * digit_size,:] = digit

    plt.figure(figsize=(10, 10))
    # # plt.imshow(figure, cmap='Greys_r')
    #
    plt.imshow(figure)
    plt.savefig(file)
    plt.show()


# =========================Gradient descent method to find the geodecy.===================================================
def get_decoded(xi):
    #xi  is the latent variable vector. 4 d in this case.
    #results in a flat origianl image version.
    z_sample = np.array([[xi]])
    z_sample = np.tile(z_sample, batch_size).reshape(batch_size, latent_dim)
    x_decoded = generator.predict(z_sample, batch_size=batch_size)
    digit = x_decoded[0].reshape(shrink * shrink * 3)
    return digit
def get_decoded2(xi):
    #xi  is the latent variable vector. 4 d in this case.
    #decoded in a matrix version
    z_sample = np.array([[xi]])
    z_sample = np.tile(z_sample, batch_size).reshape(batch_size, latent_dim)
    x_decoded = generator.predict(z_sample, batch_size=batch_size)
    digit = x_decoded[0].reshape(shrink , shrink , 3)
    return digit
from numpy import linalg as LA


# gradient descent
def gd_points(points,num_p):
    "apply the gradient deacent method to the series of poinits in the latent space, metrics in the genrated imgaes."
    lb=[]
    l=0
    for i in range(num_p-1):
        #calculate the length of the route.

        l += LA.norm(get_decoded(points[i])-get_decoded(points[i+1]))
    lb.append(l)
    for iter in range(50):

        for j in range(num_p-2):
            #Perform gradient descent
            h=0.00001
            lb_1 = LA.norm(get_decoded(points[j])-get_decoded(points[j+1]+h))+LA.norm(get_decoded(points[j+2])-get_decoded(points[j+1]+h))#evaluate function at x+h position
            lb_2 = LA.norm(get_decoded(points[j])- get_decoded(points[j+1]-h)) + LA.norm(get_decoded(points[j+2]) - get_decoded(points[j+1]-h))  # evaluate function at x-h position
            gradient=(lb_1 -lb_2)/(2*h)
            s=0
            for jj in range(num_p-1):
            #calculate the length of the route.
                s += LA.norm(get_decoded(points[jj])-get_decoded(points[jj+1]))

            print(gradient)

            # if LA.norm(get_decoded(points[j])-get_decoded(points[j+1]-0.0001 *gradient)) < s/(0.5*num_p and LA.norm(get_decoded(points[j+2])-get_decoded(points[j+1]-0.0001 *gradient))< s/num_p:
            points[j+1] -= 0.0001 *gradient

        l=0
        for ii in range(num_p-1):
            #calculate the length of the route.
            l += LA.norm(get_decoded(points[ii])-get_decoded(points[ii+1]))
        lb.append(l)



    plt.plot(lb,'ro')
    plt.savefig('gradients.png')
    # plt.show()
    return points#the points updated by GD in the latent space, mesured by the loss on generated images.
def sample(dimension,N,radius):
    'sample points within certain space around 0'
    Y=np.random.normal(size=(dimension, N))
    u=np.random.normal(N)
    r=radius*u**(1/dimension)
    X=r*Y/np.sqrt(np.sum(Y**2,axis=0))
    return X.T
def sert_points(points):

    ' add  points if interval between points on the 60*60 manifold is too large.'
    num_p=points.shape[0]
    l=0
    for ii in range(num_p-1):
        #calculate the length of the route.
        l += LA.norm(get_decoded(points[ii])-get_decoded(points[ii+1]))
    for i in range(num_p-1):
        # if LA.norm(get_decoded(points[i])-get_decoded(points[i+1])) > l/(0.4*num_p):
        if i==8:


            # # seek a points near neighbor that makes the length almost equal to both bound points.
            # #sample 10 points around the middle points, calculate the difference between the pi,pi+1, and pi+1,pi+2
            # #choose the smallest one.
            point_middle=(points[i]+points[i+1])/2
            points_=sample(points.shape[1],50,1)+point_middle#a 10*4 matrix
            #calculate the difference.
            points_dif1=LA.norm(points_-points[i], axis=1)# a 10*1 matrix
            points_dif2=LA.norm(points_-points[i+1], axis=1)# a 10*1 matrix
            points_dif=abs(points_dif1-points_dif2)
            #return the max index
            index=np.argmin(points_dif)
            # print(points)
            points= np.insert(points,i+1,points_[index],0)
            print(points.shape)


            #insert more points.
            num_p_=10

            points_ = (points[i+1] - points[i]) * np.linspace(0, 1, num_p_)[:, None] + points[i] # A (num_p, 4) array of points
            # insert this array into the original points array

            # apply the Gradient Descent.
            points_=gd_points(points_,num_p_)


            points= np.insert(points,i+1,points_[1:9],0)
            points_ = (points[i+num_p_-1] - points[i+num_p_]) * np.linspace(0, 1, num_p_)[:, None] + points[i] # A (num_p, 4) array of points
            # insert this array into the original points array

            # apply the Gradient Descent.
            points_=gd_points(points_,num_p_)


            points= np.insert(points,i+num_p_,points_[1:9],0)

    return points




def var_poins(num_p,points,var):
    "var: to the extent the loss are increased.1-10"

    lb=[]
    l=0
    for j in range(num_p-1):
        #calculate the length of the route.
        l += LA.norm(get_decoded(points[j])-get_decoded(points[j+1]))
    lb.append(l)
    for iter in range(var*10):

        for j in range(num_p-2):
            #Perform gradient descent
            h=0.0001
            lb_1 = LA.norm(get_decoded(points[j])-get_decoded(points[j+1]+h))+LA.norm(get_decoded(points[j+2])-get_decoded(points[j+1]+h))#evaluate function at x+h position
            lb_2 = LA.norm(get_decoded(points[j])- get_decoded(points[j+1]-h)) + LA.norm(get_decoded(points[j+2]) - get_decoded(points[j+1]-h))  # evaluate function at x-h position
            gradient=(lb_1 -lb_2)/(2*h)

            #control the interval of each pair not too large.
            s=0
            for j in range(num_p-1):
            #calculate the length of the route.
                s += LA.norm(get_decoded(points[j])-get_decoded(points[j+1]))
            if 0.0001*gradient < s/10:

                points[j+1] -= 0.0001*gradient
        l=0
        for j in range(num_p-1):
            #calculate the length of the route.
            l += LA.norm(get_decoded(points[j])-get_decoded(points[j+1]))
        lb.append(l)

    plt.plot(lb,'ro')
    plt.savefig('var.png')
    plt.show()
    return points




#=============  Nearest images in the training collection.=========
def route_reality(file,original_img_size,shrink,points):
    x_train=[]
    for i in range(1,1001):
        image = Image.open('/home/petered/projects/Auto-encoder/pictures/'+ str(i) +'.jpg')
        image = image.resize((shrink, shrink))  # downing sampling
        image=np.asarray(image)

        x_train.append(image)
    x_train=np.asarray(x_train)
    x_train = x_train.astype('float32') / 255.
    x_train = x_train.reshape((x_train.shape[0],) + original_img_size)

#  points[i] is a 4 d vector in the latent space.
#  POINTS[i] is a 60*60*3 matrix of the real training images.

    POINTS=np.zeros((points.shape[0],shrink,shrink,3))
    points_index=np.zeros((points.shape[0],))
    print(POINTS.shape,'check POINTS shape')
    num_p=points.shape[0]
    for i in range(num_p):
        # generated the images\
        g=get_decoded2(points[i])
        L=10000
        for j,t in enumerate(x_train):
            l=LA.norm(g-x_train[j])
            if l<L:#seeking for the nearest ones
                L=l
                POINTS[i]=x_train[j]
                points_index[i]=j
    print(points_index+1,'Nearest points index')
    #============== visualising the route on real map.===============================================================
    digit_size = shrink
    import math
    figure = np.zeros((digit_size*int(math.ceil(POINTS.shape[0]/10)+1), digit_size*10,3))
    print(int(math.ceil(POINTS.shape[0]/10)))
    j=0
    for i, xi in enumerate(POINTS):
        if i in [10,20,30,40,50,60,70,80,90,100]:
            j += 1

        digit = xi
        if i>99:
            i=i-100
        elif i>89:
            i=i-90
        elif i>79:
            i=i-80
        elif i>69:
            i=i-70
        elif i>59:
            i=i-60
        elif i>49:
            i=i-50
        elif i>39:
            i=i-40
        elif i>29:
            i=i-30
        elif i > 19:
            i=i-20
        elif i>9:
            i=i-10
        # print(i)

        figure[j * digit_size: (j + 1) * digit_size,
               i * digit_size: (i + 1) * digit_size,:] = digit

    plt.figure(figsize=(10, 10))
    # # plt.imshow(figure, cmap='Greys_r')
    #
    plt.imshow(figure)
    plt.savefig('passway_real_map_'+file+'.png')
    plt.show()

#     calculating the loss for this route.
    l=0

    for j in range(num_p-1):
#       #calculate the length of the route.
        l += LA.norm(POINTS[j]-POINTS[j+1])

    print(l,file+': latent space nearest to reality length.')

# route_reality(original_img_size,shrink,points)
# ============================================================================================


#2.  select a route by hand and calculate the loss.
# in order to check if in reality the road has shorter length the straight line
def route_manually(num_p,shrink,file='manually selected'):#file should
# select 50 images from 1-300, included.
#     n_digit=np.linspace(1, 300, num=num_p)#equal interval, silly method.
    n_digit=[1,2,5,6,7,8,9, 10,15,20,21,22,23,24,25,30,39,43,47,150,156,154,158,163,171, 172,173,174,175,179,   210,215,216,217,218,219,220,233,227,231,255,259, 267,269,  271,279,285,290,295,299]
    # n_digit=n_digit.astype(int)
    digits=[]

    for i in n_digit:
        # image = Image.open('/home/kaixin/vae-map/vae_map/pictures/'+ str(i) +'.jpg')
        image = Image.open('/home/petered/projects/Auto-encoder/pictures/'+ str(i) +'.jpg')
        # image = Image.open('/home/kaixin/vae-map/vae_map/pictures/' + str(i) + '.jpg')
        # image = scipy.misc.imread('/home/petered/projects/Auto-encoder/pictures/'+ str(i) +'.jpg')

        image = image.resize((60, 60))  # downing sampling
        image=np.asarray(image)
        # print(image.shape)
        # plt.imshow(image)
        # plt.show()
        digits.append(image)
    digits=np.asarray(digits)
    digits = digits.astype('float32') / 255.
    digits = digits.reshape((digits.shape[0],) + original_img_size)

    l=0
    print(digits.shape)
    for j in range(num_p-1):
#       #calculate the length of the route.
        l += LA.norm(digits[j]-digits[j+1])

    print(l,file+' length.')
#=====visualise the manuallly selected ones.=====
    digit_size = shrink
    figure = np.zeros((digit_size*5, digit_size*10,3))
    j=0
    for i, xi in enumerate(digits):
        if i in [10,20,30,40,50]:
            j += 1

        digit = xi
        if i>39:
            i=i-40
        elif i>29:
            i=i-30
        elif i > 19:
            i=i-20
        elif i>9:
            i=i-10
        # print(i)

        figure[j * digit_size: (j + 1) * digit_size,
               i * digit_size: (i + 1) * digit_size,:] = digit

    plt.figure(figsize=(10, 10))
    # # plt.imshow(figure, cmap='Greys_r')
    #
    plt.imshow(figure)
    plt.savefig('passway_'+file+'real frames.png')
    plt.show()
    return digits #the same structure and shape as training
# ============================================================================================================

start=1
destination=300
image_s=[]

image = Image.open('/home/petered/projects/Auto-encoder/pictures/'+ str(start) +'.jpg')
image = image.resize((shrink, shrink))
image=np.asarray(image)
image=image.tolist()
image_s.append(image)
image_s=np.asarray(image_s)
image_s= np.tile(image_s, ( batch_size,1,1,1))

image_d=[]
image = Image.open('/home/petered/projects/Auto-encoder/pictures/'+ str(destination) +'.jpg')
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

points = (z_d - z_s) * np.linspace(0, 1, num_p)[:, None] + z_s # A (num_p, 4) array of points
print(points[1],'checking producing straight line points in latent space')

# =================visualising generated images route=======================================
# visualise_route('passway_straight.png',points,shrink,batch_size,latent_dim)
# print(points.shape)
# sert_points(num_p,points)
# =======METHOD PART=========================================================
# points= gd_points(points,num_p)
# route_reality('try1',original_img_size,shrink,points)

# for i in range(1):#for insert a randomly selected points.
#     points=sert_points(points)
# print(points.shape)


# points=var_poins(num_p , points ,10)



#======================= visualising the route on generated images==============================

# visualise_route('passway_dg.png',points,shrink,batch_size,latent_dim)

# # ==== visualising the route with real frames according to generated images===========================================
# route_reality('bias',original_img_size,shrink,points)#very risky, gradient ascend.
# route_reality('try2',original_img_size,shrink,points)
# route_reality('straight_line',original_img_size,shrink,points)

#===== select a route by hand and calculate the loss.===================================
# hard coded. visualisation included.
digits=route_manually(num_p,shrink,file='manually selected')
# feed them through encoder
digits_encoded= encoder.predict(digits, batch_size=batch_size)
# feed them through the decoder
visualise_route('reconstructed_from_manually',digits_encoded,shrink,batch_size,latent_dim)





