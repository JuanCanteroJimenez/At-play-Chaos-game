import glob
import runai.ga.keras
import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import tensorflow_probability as tfp
import time
from sklearn.decomposition import PCA
from tensorflow import keras



tf.keras.mixed_precision.set_global_policy('mixed_float16')


filelist =  glob.glob('data/caos_images/*.png')
def batch_images(files):
    result = []
    for fname in files:
        pre = tf.keras.utils.img_to_array(tf.keras.utils.load_img(fname, color_mode="grayscale"))
        
        if pre.shape == (64, 64, 1):
            result.append(pre)
    return(np.array(result))
print(int(np.rint(len(filelist)*0.8)))
train_images = batch_images(filelist[0:int(np.rint(len(filelist)*0.8))])
test_images = batch_images(filelist[int(np.rint(len(filelist)*0.8)):(len(filelist)-1)])
print(train_images.shape)
print(test_images.shape)
def preprocess_images(images):
  images = images.reshape((images.shape[0], 64, 64, 1)) / 255.
  return np.where(images > .5, 1.0, 0.0).astype('float32')

train_images = preprocess_images(train_images)
test_images = preprocess_images(test_images)

train_size = train_images.shape[0]
batch_size = 1
test_size = test_images.shape[0]

train_dataset = (tf.data.Dataset.from_tensor_slices(train_images)
                 .shuffle(train_size).batch(batch_size))
test_dataset = (tf.data.Dataset.from_tensor_slices(test_images)
                .shuffle(test_size).batch(batch_size))




class ResidualUnit(keras.layers.Layer):
    def __init__(self, filters, strides=1, activation="relu", **kwargs):
        super().__init__(**kwargs)
        #self.activation = keras.activations.get(activation)
        self.main_layers = [
            keras.layers.Conv2D(filters, 3, strides=strides,
                padding="same", use_bias=False),
            keras.layers.BatchNormalization(),
            keras.layers.Activation(activation, dtype="float32"),
            keras.layers.Conv2D(filters, 3, strides=1,
                padding="same", use_bias=False),
            keras.layers.BatchNormalization()]
        self.skip_layers = []
        if strides == [2,2]:
            self.skip_layers = [
                keras.layers.Conv2D(filters, 1, strides=strides,
                    padding="same", use_bias=False),
                keras.layers.BatchNormalization()]
    def call(self, inputs):
        Z = inputs
        activa = keras.layers.ReLU(dtype="float32")
        for layer in self.main_layers:
            Z = layer(Z)
        skip_Z = inputs
        for layer in self.skip_layers:
            skip_Z = layer(skip_Z)
        return activa(Z+skip_Z)


class ResidualUnit_t(keras.layers.Layer):
    def __init__(self, filters, strides=1, activation="relu", **kwargs):
        super().__init__(**kwargs)
        #self.activation = keras.activations.get(activation)
        self.main_layers = [
            keras.layers.Conv2DTranspose(filters, 3, strides=strides,
                padding="same", use_bias=False),
            keras.layers.BatchNormalization(),
            keras.layers.Activation(activation, dtype="float32"),
            
            keras.layers.Conv2DTranspose(filters, 3, strides=1,
                padding="same", use_bias=False),
            keras.layers.BatchNormalization()]
        self.skip_layers = []
        if strides == [2,2]:
            self.skip_layers = [
                keras.layers.Conv2DTranspose(filters, 1, strides=strides,
                    padding="same", use_bias=False),
                keras.layers.BatchNormalization()]
    def call(self, inputs):
        Z = inputs
        activa = keras.layers.ReLU(dtype="float32")

        for layer in self.main_layers:
            Z = layer(Z)
        skip_Z = inputs
        for layer in self.skip_layers:
            skip_Z = layer(skip_Z)
        return activa(Z + skip_Z)

    









class CVAE(tf.keras.Model):
  """Convolutional variational autoencoder."""

  def __init__(self, latent_dim):
    super(CVAE, self).__init__()
    self.latent_dim = latent_dim
    self.encoder = tf.keras.Sequential()
    self.encoder.add(tf.keras.layers.InputLayer(input_shape=(64, 64, 1)))
    #for filters in [512] * 3 + [256] * 3 + [128] * 4 + [64] * 3 + [32] * 6:
    prev_filters = 64
    for filters in  [64] * 3 + [128] * 3 + [256] * 3 + [512] * 3:
        strides = [1,1] if filters == prev_filters else [2,2]
        self.encoder.add(ResidualUnit(filters, strides=strides))
        prev_filters = filters
    
    self.encoder.add(tf.keras.layers.Flatten())
    
    self.encoder.add(tf.keras.layers.Dense(latent_dim + latent_dim))
    self.encoder.add(tf.keras.layers.Activation("linear",dtype="float32"))
       
    
    self.decoder = tf.keras.Sequential()
    self.decoder.add(tf.keras.layers.Dense(4*4*128, use_bias = False, input_shape=(latent_dim,)))
    
    
    self.decoder.add(tf.keras.layers.Reshape((4, 4, 128)))
    
    prev_filters = 128
    #for filters in [64] * 3 + [128] * 4 + [256] * 6 + [512] * 3:
    for filters in [512] * 3 + [256] * 3 + [128] * 3 + [64] * 3 :
        strides = [1,1] if filters == prev_filters else [2,2]
        self.decoder.add(ResidualUnit_t(filters, strides=strides))
        prev_filters = filters

    self.decoder.add(keras.layers.Conv2DTranspose(1, (3, 3), strides=1, padding='same', use_bias=False))
    self.decoder.add(tf.keras.layers.Activation("linear",dtype="float32"))




    
    


  @tf.function
  def sample(self, eps=None):
    if eps is None:
      eps = tf.random.normal(shape=(100, self.latent_dim))
    return self.decode(eps, apply_sigmoid=True)

  def encode(self, x):
    mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
    return mean, logvar

  def reparameterize(self, mean, logvar):
    eps = tf.random.normal(shape=mean.shape,dtype="float32")
    return eps * tf.exp(logvar * tf.constant(0.5, dtype="float32")) + mean

  def decode(self, z, apply_sigmoid=False):
    logits = self.decoder(z)
    if apply_sigmoid:
      probs = tf.sigmoid(logits)
      return probs
    return logits

optimizer = tf.keras.optimizers.Adam(1e-4)
optimizer = runai.ga.keras.optimizers.Optimizer(optimizer,steps=32)



def log_normal_pdf(sample, mean, logvar, raxis=1):
  log2pi = tf.math.log(2. * np.pi)
  return tf.reduce_sum(
      -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
      axis=raxis)


def compute_loss(model, x):
  mean, logvar = model.encode(x)
  z = model.reparameterize(mean, logvar)
  x_logit = model.decode(z)
  cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
  logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
  logpz = log_normal_pdf(z, 0., 0.)
  logqz_x = log_normal_pdf(z, mean, logvar)
  return -tf.reduce_mean(logpx_z + logpz - logqz_x)


@tf.function
def train_step(model, x, optimizer):
  """Executes one training step and returns the loss.

  This function computes the loss and gradients, and uses the latter to
  update the model's parameters.
  """
  with tf.GradientTape() as tape:
    loss = compute_loss(model, x)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))


epochs = 500
# set the dimensionality of the latent space to a plane for visualization later
latent_dim = 100
num_examples_to_generate = 1

# keeping the random vector constant for generation (prediction) so
# it will be easier to see the improvement.
random_vector_for_generation = tf.random.normal(
    shape=[num_examples_to_generate, latent_dim])
model = CVAE(latent_dim)

def generate_and_save_images(model, epoch, test_sample):
  mean, logvar = model.encode(test_sample)
  z = model.reparameterize(mean, logvar)
  
  predictions = model.sample(z)
  fig = plt.figure(figsize=(4, 4))

  for i in range(predictions.shape[0]):
    plt.subplot(4, 4, i + 1)
    plt.imshow(predictions[i, :, :, 0], cmap='gray')
    plt.axis('off')

  # tight_layout minimizes the overlap between 2 sub-plots
  plt.savefig('images_at_epoch_{:04d}.png'.format(epoch))
  plt.show()
  plt.close()



def generate_scatter(model, epoch, test_sample):
  mean, logvar = model.encode(test_sample)
  z = model.reparameterize(mean, logvar)
  pca = PCA(n_components=2)
  encoded_2dd = pca.fit_transform(z)
    
  fig = plt.figure(figsize=(4, 4))

    
  
  plt.scatter(encoded_2dd[:, 0], encoded_2dd[:, 1], cmap='gray',s=0.1, alpha=0.5)
  plt.axis('off')

  # tight_layout minimizes the overlap between 2 sub-plots
  plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
  plt.show()
  plt.close()


assert batch_size >= num_examples_to_generate
for test_batch in test_dataset.take(1):
  test_sample = test_batch[0:num_examples_to_generate, :, :, :]


for epoch in range(1, epochs + 1):
  start_time = time.time()
  for train_x in train_dataset:
    train_step(model, train_x, optimizer)
  end_time = time.time()

  loss = tf.keras.metrics.Mean()
  for test_x in test_dataset:
    loss(compute_loss(model, test_x))
  elbo = -loss.result()
  
  print('Epoch: {}, Test set ELBO: {}, time elapse for current epoch: {}'
        .format(epoch, elbo, end_time - start_time))
  generate_scatter(model, epoch, test_images)
  generate_and_save_images(model, epoch, test_sample)
