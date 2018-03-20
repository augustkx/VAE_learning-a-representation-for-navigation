'''This script demonstrates how to learn a representaion of image data using a variational autoencoder with Keras and convolution layers.

The structure draw much inspiratin from https://github.com/keras-team/keras
'''

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

filters =32# number of convolutional filters to use
num_conv = 3
stri=1 #stride value of last con layer
stri2=1

batch_size = 20
if K.image_data_format() == 'channels_first':
    original_img_size = (img_chns, img_rows, img_cols)
else:
    original_img_size = (img_rows, img_cols, img_chns)
latent_dim = 4
intermediate_dim = 256
epsilon_std = 1.0
epochs = 20000

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
decoder_upsample = Dense(filters * (shrink/2) * (shrink/2), activation='relu')

if K.image_data_format() == 'channels_first':
    
    output_shape = (batch_size, filters, (shrink/2),(shrink/2))
else:
    
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
from keras import optimizers
# optimizer=optimizers.SGD(lr=0.0001, momentum=0.0, decay=0.0, nesterov=False)
optimizer=optimizers.RMSprop(lr=0.0001, rho=0.9, epsilon=1e-07, decay=0.999)
vae.compile(optimizer=optimizer, loss= None)
# vae.compile(optimizer='rmsprop', loss=None)
vae.summary()



#========================load images for traing==========================================
x_train=[]
for i in range(1,1001):
    # image = Image.open('/home/kaixin/vae-map/vae_map/pictures/'+ str(i) +'.jpg')
    image = Image.open('/home/petered/projects/Auto-encoder/pictures/'+ str(i) +'.jpg')
    # image = Image.open('/home/kaixin/vae-map/vae_map/pictures/' + str(i) + '.jpg')
    # image = scipy.misc.imread('/home/petered/projects/Auto-encoder/pictures/'+ str(i) +'.jpg')
    image = image.resize((shrink, shrink))  # downing sampling
    image=np.asarray(image)
    # image= np.mean(image, axis=(2))

    x_train.append(image)
x_train=np.asarray(x_train)



x_train = x_train.astype('float32') / 255.
x_train = x_train.reshape((x_train.shape[0],) + original_img_size)


print('x_train.shape:', x_train.shape)

early_stopping = EarlyStopping(monitor='val_loss', patience=500)
train_history=vae.fit(x_train,
        callbacks=[early_stopping],
        shuffle=True,
        epochs=epochs,
        batch_size=batch_size)

#=============================plotting loss during training
loss = train_history.history['loss']
plt.plot(loss)
plt.savefig('loss_conv-4d.png')
plt.show()

vae.save('vae_conv-4d.h5')
from keras.utils import plot_model
plot_model(vae, to_file='model-4d.png')

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




# =================display a 2D manifold of the digits============
n =15  # figure with 15x15 digits
digit_size = shrink
figure = np.zeros((digit_size * n, digit_size * n,3))
# linearly spaced coordinates on the unit square were transformed through the inverse CDF (ppf) of the Gaussian
# to produce values of the latent variables z, since the prior of the latent space is Gaussian
grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi, yi,1,1]])#under the asumption of 4 dimension latent space
        z_sample = np.tile(z_sample, batch_size).reshape(batch_size, latent_dim)
        x_decoded = generator.predict(z_sample, batch_size=batch_size)

        digit = x_decoded[0].reshape(digit_size, digit_size,3)
        figure[i * digit_size: (i + 1) * digit_size,
               j * digit_size: (j + 1) * digit_size,:] = digit

plt.figure(figsize=(10, 10))
plt.imshow(figure)
plt.savefig('manifold-4d.png')
plt.show()
