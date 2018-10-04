
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
from keras.models import model_from_json

from numpy import linalg as LA
import math
import seaborn as sns

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

decoder=generator
def vis():
    n =15  # figure with 15x15 digits
    digit_size = shrink
    figure = np.zeros((digit_size * n, digit_size * n,3))
    # linearly spaced coordinates on the unit square were transformed through the inverse CDF (ppf) of the Gaussian
    # to produce values of the latent variables z, since the prior of the latent space is Gaussian
    grid_x = norm.ppf(np.linspace(-1,1, n))
    grid_y = norm.ppf(np.linspace(-1, 1, n))

    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z_sample = np.array([[xi, yi,1.5,5]])#under the asumption of 4 dimension latent space
            z_sample = np.tile(z_sample, batch_size).reshape(batch_size, latent_dim)
            x_decoded = generator.predict(z_sample, batch_size=batch_size)

            digit = x_decoded[0].reshape(digit_size, digit_size,3)
            figure[i * digit_size: (i + 1) * digit_size,
                   j * digit_size: (j + 1) * digit_size,:] = digit

    plt.figure(figsize=(10, 10))
    # plt.imshow(figure, cmap='Greys_r')

    plt.imshow(figure)
    plt.savefig('manifold-4d.png')
    plt.show()







'''gradient descent method to update a path.'''
def get_decoded(xi):
    #xi  is the latent variable vector. 4 d in this case.
    #results in a flat origianl image version.
    z_sample = np.array([[xi]])
    z_sample = np.tile(z_sample, batch_size).reshape(batch_size, latent_dim)
    x_decoded = decoder.predict(z_sample, batch_size=batch_size)
    digit = x_decoded[0].reshape(shrink * shrink * 3)
    return digit
def get_decoded2(xi):
    #xi  is the latent variable vector. 4 d in this case.
    #decoded in a matrix version
    z_sample = np.array([[xi]])
    z_sample = np.tile(z_sample, batch_size).reshape(batch_size, latent_dim)
    x_decoded = decoder.predict(z_sample, batch_size=batch_size)
    digit = x_decoded[0].reshape(shrink , shrink , 3)
    return digit

def gd_points(points,num_p):
    "apply the gradient deacent method to the series of poinits in the latent space, metrics in the genrated imgaes."
    lb=[]
    l=0
    for i in range(num_p-1):
        #calculate the length of the route.

        l += LA.norm(get_decoded(points[i])-get_decoded(points[i+1]))
    lb.append(l)
    for iter in range(20):
        print('iter',iter)
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
    return points
    #the points updated by GD in the latent space, measured by the loss on generated images.





'''visualize a route from a path in the latent space.'''
def visualise_route(file,points,shrink,batch_size,num_p):
    #file: is a string of the name of output image.
    #visualise from the latent space points to 60*60 reality.
    #fixed 50 images.

    digit_size = shrink
    figure = np.zeros((digit_size*num_p/10, digit_size*10,3))
    j=0
    POINTS=[]
    for i, xi in enumerate(points):
        if i in range(10,120,10):
            j += 1
        z_sample = np.array([[xi]])
        z_sample = np.tile(z_sample, batch_size).reshape(batch_size, latent_dim)
        x_decoded = decoder.predict(z_sample, batch_size=batch_size)
        digit = x_decoded[0].reshape(digit_size, digit_size,3)
        i = i - int(math.floor(i / 10))*10

        figure[j * digit_size: (j + 1) * digit_size,
               i * digit_size: (i + 1) * digit_size,:] = digit
        POINTS.append(digit)
    plt.figure(figsize=(10, 10))
    plt.imshow(figure)
    plt.savefig(file)
    # plt.show()
    return POINTS



'''Route of nearest images in the training collection , w.r.t reconstrucgted image route'''
def route_reality(file,shrink,points):
    x_train=[]
    for i in range(1,1001):
        image = Image.open('/home/petered/projects/Auto-encoder/pictures/'+ str(i) +'.jpg')
        # image = Image.open('/home/kaixin/vae-map/vae_map/pictures_new/' + str(i) + '.jpg')
        #
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
    figure = np.zeros((digit_size*int(math.ceil(POINTS.shape[0]/10)), digit_size*10,3))
    print(int(math.ceil(POINTS.shape[0]/10)))
    j=0
    for i, xi in enumerate(POINTS):
        if i in range(10,110,10):
            j += 1
        digit = xi
        i=i-int(math.floor(i/10))*10
         # print(i)

        figure[j * digit_size: (j + 1) * digit_size,
               i * digit_size: (i + 1) * digit_size,:] = digit

    plt.figure(figsize=(10, 10))
    plt.imshow(figure)
    plt.savefig('passway_real_map_'+file+'.png')
    # plt.show()


#     calculating the loss for this route.
    l=0

    for j in range(num_p-1):
#       #calculate the length of the route.
        l += LA.norm(POINTS[j]-POINTS[j+1])

    print(l,file+': latent space nearest to reality length.')
    return POINTS

'''manually select a route
 in order to check if in reality the road has shorter length the straight line.'''
def route_manually(num_p,shrink,file='manually selected'):#file should
# select 50 images from 1-300, included.
#     n_digit=np.linspace(1, 300, num=num_p)#equal interval, silly method.
    #case one
    # n_digit=[1,2,5,6,7,8,9, 10,15,20,21,22,23,24,25,30,39,43,47,150, 156,154,158,163,171, 172,173,174,175,179, 210,215,216,217,218,219,220,233,227,231,255,259, 267,269,271,279,285,290,295,299]
    # n_digit=[1,5,8,9, 10,15,20,21,22,23,24,25,30,31,32,33,34,35,36,45,175,179,181,182 ,186,190,192,195, 200,205,  210,215,216,217,218,219,220,233,227,231,255,259, 267,269,  271,279,285,290,295,299]
#     n_digit=[300,301,309,312,319,324,326, 327,330,338,341,349,354,360,368,369,516,520,524,530,535,539,575,576,580, 583,585,588,590,592,  594,598,599,600,602,605,610,615,620,622,628,630,633,638, 640,642,646,647,649,650]
#     n_digit=[300,301,309,312,319,324,326, 327,330,338,341,349,354,360,368,369,516,520,524,530,535,539,575,576,580, 583,585,588,590,592,  594,598,599,600,602,605,610,615,620,622,628,630,633,638, 640,642,646,647,649,650]
    n_digit=[300,301,309,312,319,324,326, 327,330,338,341,349,354,360,368,369,516,520,524,530,535,539,575,576,580, 583,585,588,590,592,  594,598,599,600,602,605,610,615,620,624,719,720,722,725, 730,735,740,745,747,750]
#     n_digit=[1,9, 10,15,20,21,25,30,31,32,36,40,45,49,60,70,80,90,100,120,130,140,150 ,160,170,180,190,195, 200,205,  210,215,216,217,218,219,220,233,227,231,255,259, 267,269,  271,279,285,290,295,299]
#     n_digit=[1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46, 49, 52, 55, 58, 61, 64, 67, 70, 73, 76, 79, 82, 85, 88, 91, 94, 97, 100, 103, 106, 109, 112, 115, 118, 121, 124, 127, 130, 133, 136, 139, 142, 145, 148, 151, 154, 157, 160, 163, 166, 169, 172, 175, 178, 181, 184, 187, 190, 193, 196, 199, 202, 205, 208, 211, 214, 217, 220, 223, 226, 229, 232, 235, 238, 241, 244, 247, 250, 253, 256, 259, 262, 265, 268, 271, 274, 277, 280, 283, 286, 289, 292, 295, 298]
    #case two


    #case three



    # n_digit=n_digit.astype(int)
    digits=[]

    for i in n_digit:
        # image = Image.open('/home/petered/projects/Auto-encoder/
        # tures/'+ str(i) +'.jpg')
        # image = Image.open('/home/kaixin/vae-map/vae_map/pictures_new/' + str(i) + '.jpg')
        image = Image.open('/home/petered/projects/Auto-encoder/pictures/'+ str(i) +'.jpg')
        image = image.resize((60, 60))  # downing sampling
        image=np.asarray(image)

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
#=====visualise the ground truth / manuallly selected ones.=====
    digit_size = shrink
    figure = np.zeros((digit_size*num_p/10, digit_size*10,3))
    j=0
    for i, xi in enumerate(digits):
        if i in range(10,110,10):
            j += 1
        digit = xi
        i = i - int(math.floor(i / 10))*10

        figure[j * digit_size: (j + 1) * digit_size,
               i * digit_size: (i + 1) * digit_size,:] = digit

    plt.figure(figsize=(10, 10))
    plt.imshow(figure)
    plt.savefig('passway_'+file+'real frames.png')
    # plt.show()
    return digits #the same structure and shape as training



''' evaluate each reconstrcted image in the route w.r.t manually selected one, by measuring the distance of it and the next image.
    Compare the distance with that of the image and a random image.
    for the reconstructed image in the route produced w.r.t the ground truth route.'''
def metric(points,shrink,batch_size):

    digit_size = shrink
    figure = np.zeros((digit_size*5, digit_size*10,3))
    j=0
    route=[]
    for i, xi in enumerate(points):


        if i in range(10,110,10):
            j += 1

        z_sample = np.array([[xi]])

        z_sample = np.tile(z_sample, batch_size).reshape(batch_size, latent_dim)

        x_decoded = generator.predict(z_sample, batch_size=batch_size)

        digit = x_decoded[0].reshape(digit_size, digit_size,3)
        route.append(digit)

    x_train=[]
    for i in range(1,1001):
        image = Image.open('/home/petered/projects/Auto-encoder/pictures/'+ str(i) +'.jpg')
        # image = Image.open('/home/kaixin/vae-map/vae_map/pictures_new/' + str(i) + '.jpg')

        image = image.resize((shrink, shrink))  # downing sampling
        image=np.asarray(image)

        x_train.append(image)
    x_train=np.asarray(x_train)
    x_train = x_train.astype('float32') / 255.
    x_train = x_train.reshape((x_train.shape[0],) + original_img_size)
    d=[]
    d_latent=[]
    #route is a vector consisting of the generated images.
    for i in range(0,len(route)-1):
        d_n=LA.norm(route[i]-route[i+1])
        #randomly select a image from the traning set.
        d_r=LA.norm(route[i]- x_train[np.random.randint(0,1000)])
        d.append(d_r-d_n)

        d_n_latent=LA.norm(points[i]-points[i+1])
        #randomly select a image from the traning set, and feed it through encoder
        image_d = []
        image = Image.open('/home/petered/projects/Auto-encoder/pictures/' + str(np.random.randint(1,1001)) + '.jpg')
        image = image.resize((shrink, shrink))
        image = np.asarray(image)
        image = image.tolist()
        image_d.append(image)
        image_d = np.asarray(image_d)
        image_d = np.tile(image_d, (batch_size, 1, 1, 1))
        image_d = image_d.astype('float32') / 255.
        image_d = image_d.reshape((image_d.shape[0],) + original_img_size)
        d_r_latent=LA.norm( points[i]-encoder.predict(image_d, batch_size=batch_size)[0])
        d_latent.append(d_r_latent-d_n_latent)



    sns.distplot(d)
    plt.savefig('distance.png')#compare the route image within a selected route. no use in the thesis.
    plt.clf()
    sns.distplot(d_latent)
    plt.savefig('distance_latent.png')#compare the points in latent space w.r.t the manually selected route.
    plt.clf()



    sns.distplot(d)
    plt.savefig('distance.png')
    plt.clf()
    sns.distplot(d_latent)
    plt.savefig('distance_latent.png')
    plt.clf()

    '''evaluate how good a produce route is, w.r.t the manually selected one.

    '''

def evaluate(points,digits_encoded,POINTS,digits,shrink,batch_size):
    #POINTS are the route w.r.t the producced route with the method in this work.
    #digits are the route selected manually.
    dis_produced=0
    dis_manually=0
    max_p=0
    max_m=0
    d_p=[]
    d_m=[]
    d_p_latent=[]
    d_m_latent=[]
    for i in range(len(POINTS)-1):
        dis_produced += abs(LA.norm(POINTS[i]-POINTS[i+1]))
        d_p.append( abs(LA.norm(POINTS[i]-POINTS[i+1])))
        d_p_latent.append(abs(LA.norm(points[i]-points[i+1])))
        if abs(LA.norm(POINTS[i]-POINTS[i+1]))> max_p:
            max_p=abs(LA.norm(POINTS[i]-POINTS[i+1]))

        dis_manually += abs(LA.norm(digits[i]-digits[i+1]))
        d_m.append(abs(LA.norm(digits[i]-digits[i+1])))
        d_m_latent.append(abs(LA.norm(digits_encoded[i]-digits_encoded[i+1])))
        if abs(LA.norm(digits[i]-digits[i+1]))> max_m:
            max_m=abs(LA.norm(digits[i]-digits[i+1]))
    #    the sum of distance of each neighbouring image.
    print('dis_produced, dis_manually:',dis_produced,',', dis_manually)
    #the maximium distance between each pair.
    print('max_p,max_m', max_p,max_m)

    sns.distplot(d_p,color="skyblue", label="Distribution of neighbour image distances in route produced with GD")
    # plt.savefig('distance_p.png')#distribution of distance of neighbour images(real frames) in route
    sns.distplot(d_m,color="red", label="Distribution of neighbour image distances in route manually sellected")
    plt.legend()
    plt.savefig('distance_within_route.png')#distribution of distance of neighbour images(real frames) in manually selected route
    plt.clf()
    sns.distplot(d_p_latent,color="skyblue", label="Distribution of neighbour pair distances in path produced with GD")
    plt.savefig('distance_p_latent.png')#
    plt.clf()
    sns.distplot(d_m_latent,color="red", label="Distribution of neighbour pair distances in path w.r.t. manually selected route")
    plt.legend(loc=2, fontsize = 'x-small')
    plt.savefig('distance_m_latent.png')



    return dis_produced, dis_manually
