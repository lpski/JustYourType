import tensorflow as tf, glob, matplotlib.pyplot as plt, numpy as np, os, PIL, time
from tensorflow.keras import layers, Model, losses, optimizers
from utils import get_characters, get_data
from dataset import Dataset

from IPython import display



# Loss Functions
cross_entropy = losses.BinaryCrossentropy(from_logits=True)
def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

def discriminator_loss(real_output, fake_output):
    # cross_entropy = losses.BinaryCrossentropy(from_logits=True)
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


# Model
class Model():
    buffer_size: int
    batch_size: int
    generator: Model = None
    discriminator: Model = None

    def __init__(self, buffer_size: int = 60000, batch_size: int = 256):
        self.buffer_size = buffer_size
        self.batch_size = batch_size


    # Generator Content
    def make_generator(self) -> Model:
        start_dim, end_dim = 16, 64

        model = tf.keras.Sequential()
        model.add(layers.Dense(start_dim*start_dim*256, use_bias=False, input_shape=(100,)))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Reshape((start_dim, start_dim, 256)))
        assert model.output_shape == (None, start_dim, start_dim, 256)

        model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
        assert model.output_shape == (None, start_dim, start_dim, 128)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        assert model.output_shape == (None, start_dim * 2, start_dim * 2, 64)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
        assert model.output_shape == (None, end_dim, end_dim, 1)

        return model
        

    # Discriminator Content
    def make_descriminator(self) -> Model:
        model = tf.keras.Sequential()
        model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[64, 64, 1]))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Flatten())
        model.add(layers.Dense(1))

        return model
        

    # Training
    @tf.function
    def train_step(self, images, gen_optimizer: optimizers.Adam, dis_optimizer: optimizers.Adam):
        noise_dim = 100
        noise = tf.random.normal([self.batch_size, noise_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(noise, training=True)

            real_output = self.discriminator(images, training=True)
            fake_output = self.discriminator(generated_images, training=True)

            gen_loss = generator_loss(fake_output)
            disc_loss = discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        gen_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        dis_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

    def train(self, dataset, epochs: int):
        self.generator, self.discriminator = self.make_generator(), self.make_descriminator()
        gen_opt, dis_opt = optimizers.Adam(1e-4), optimizers.Adam(1e-4)

        # Set up checkpoints
        checkpoint_dir = './checkpoints'
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        checkpoint = tf.train.Checkpoint(
            generator_optimizer=gen_opt,
            discriminator_optimizer=dis_opt,
            generator=self.generator,
            discriminator=self.discriminator
        )
        
        # Set params
        noise_dim = 100
        num_examples_to_generate = 16
        seed = tf.random.normal([num_examples_to_generate, noise_dim])

        # Start Training
        for epoch in range(epochs):
            start = time.time()

            for image_batch in dataset:
                self.train_step(image_batch, gen_opt, dis_opt)

            # Produce images for the GIF as you go
            display.clear_output(wait=True)
            self.generate_and_save_images(self.generator, epoch + 1, seed)

            # Save the model every 15 epochs
            if (epoch + 1) % 15 == 0:
                checkpoint.save(file_prefix = checkpoint_prefix)

            print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

        # Generate after the final epoch
        display.clear_output(wait=True)
        self.generate_and_save_images(self.generator, epochs, seed)

    def generate_and_save_images(self, model, epoch, test_input):
        # Notice `training` is set to False.
        # This is so all layers run in inference mode (batchnorm).
        predictions = model(test_input, training=False)

        fig = plt.figure(figsize=(4, 4))

        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i+1)
            plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
            plt.axis('off')

        plt.savefig('train_images/epoch_{:04d}.png'.format(epoch))
        # if epoch % 100 == 0: plt.show()
        # else: plt.close()






def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256)  # Note: None is the batch size

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 28, 28, 1)

    return model






if __name__ == '__main__':
    batch_size, buffer_size, epochs = 256, 60000, 500
    m = Model(buffer_size, batch_size)
    data = Dataset()
    train_X, train_y = data.encoded_fonts('AA')
    train_X = train_X.reshape(train_X.shape[0], 64, 64, 1).astype('float32')
    # train_X = train_X.reshape(train_X.shape[0], 64, 64, 1)
    train_X = (train_X - 127.5) / 127.5  # Normalize the images to [-1, 1]
    train_dataset = tf.data.Dataset.from_tensor_slices(train_X).shuffle(buffer_size).batch(batch_size)
    # train_dataset = tf.data.Dataset.from_tensor_slices(train_X).shuffle(batch_size).batch(batch_size)
    m.train(train_dataset, epochs)



