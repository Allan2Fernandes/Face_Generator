import math

import tensorflow as tf
from keras.layers import Input, Conv2D, LeakyReLU, BatchNormalization, Dense, Flatten, Reshape, Conv2DTranspose
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop, Adam
from keras.losses import BinaryCrossentropy
import time

import Visualize_data


class Network_Builder:
    def __init__(self, image_size, codings_size):
        self.generator_model = None
        self.dim1 = image_size[0]
        self.dim2 = image_size[1]
        self.discriminator_input_shape = (self.dim1, self.dim2, 3)
        self.codings_size = codings_size
        pass

    def build_generator(self):
        # It has to start with
        dimension = int(self.dim1 / math.pow(2, 5))
        self.generator_model = Sequential([
            Dense(units=(dimension * dimension * 512), input_shape=[self.codings_size]),
            Reshape((dimension, dimension, 512)),
            # Up-sample it 5 times
            BatchNormalization(),
            Conv2DTranspose(filters=512, activation='selu', strides=(2, 2), kernel_size=(4, 4), padding='same', use_bias=False),
            Conv2DTranspose(filters=512, activation='selu', strides=(1, 1), kernel_size=(4, 4), padding='same', use_bias=False),
            BatchNormalization(),
            Conv2DTranspose(filters=256, activation='selu', strides=(2, 2), kernel_size=(4, 4), padding='same', use_bias=False),
            Conv2DTranspose(filters=256, activation='selu', strides=(1, 1), kernel_size=(4, 4), padding='same', use_bias=False),
            BatchNormalization(),
            Conv2DTranspose(filters=128, activation='selu', strides=(2, 2), kernel_size=(4, 4), padding='same', use_bias=False),
            Conv2DTranspose(filters=128, activation='selu', strides=(1, 1), kernel_size=(4, 4), padding='same', use_bias=False),
            BatchNormalization(),
            Conv2DTranspose(filters=64, activation='selu', strides=(2, 2), kernel_size=(4, 4), padding='same', use_bias=False),
            Conv2DTranspose(filters=64, activation='selu', strides=(1, 1), kernel_size=(4, 4), padding='same', use_bias=False),
            BatchNormalization(),
            Conv2DTranspose(filters=3, activation='tanh', strides=(2, 2), kernel_size=(4, 4), padding='same', use_bias=False),
        ])
        pass

    def build_discriminator(self):
        self.discriminator_model = Sequential([
            Input(shape=self.discriminator_input_shape),

            Conv2D(filters=64, strides=(2, 2), kernel_size=(4, 4), padding='same', kernel_initializer='he_normal', use_bias=False),
            BatchNormalization(),
            LeakyReLU(),

            Conv2D(filters=64, strides=(2, 2), kernel_size=(4, 4), padding='same', kernel_initializer='he_normal', use_bias=False),
            BatchNormalization(),
            LeakyReLU(),

            Conv2D(filters=128, strides=(2, 2), kernel_size=(4, 4), padding='same', kernel_initializer='he_normal', use_bias=False),
            BatchNormalization(),
            LeakyReLU(),

            Conv2D(filters=128, strides=(2, 2), kernel_size=(4, 4), padding='same', kernel_initializer='he_normal', use_bias=False),
            BatchNormalization(),
            LeakyReLU(),

            Conv2D(filters=256, strides=(2, 2), kernel_size=(4, 4), padding='same', kernel_initializer='he_normal', use_bias=False),
            LeakyReLU(),

            Flatten(),
            Dense(units=1, activation='sigmoid')
        ])

        pass

    def build_GAN(self):
        self.GAN = Sequential([self.generator_model, self.discriminator_model])
        pass

    def compile_models(self):
        # Only have to compile the discriminator and GAN
        self.discriminator_optimizer = tf.keras.optimizers.RMSprop(learning_rate=.0001, clipvalue=1.0, decay=1e-8)
        self.generator_optimizer = tf.keras.optimizers.RMSprop(learning_rate=.0001, clipvalue=1.0, decay=1e-8)

        self.loss_function = BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
        pass

    def get_generator(self):
        return self.generator_model

    def get_discriminator(self):
        return self.discriminator_model

    def get_GAN(self):
        return self.GAN

    def summarize_all_models(self):
        self.generator_model.summary()
        self.discriminator_model.summary()
        self.GAN.summary()
        pass


    def train_the_network(self, dataset, epochs, codings_size):
        for epoch in range(epochs):
            start_time = time.time()
            print("Current_epoch is {}".format(epoch))
            for step, real_images_batch in enumerate(dataset):
                if step%50 == 0:
                    print("Current Step is: {}".format(step))
                batch_size = real_images_batch.shape[0]
                with tf.GradientTape() as discriminator_tape:
                    #Create noise
                    noise = tf.random.normal(shape=(batch_size, codings_size))
                    #Create fake images by passing the noise through the generator
                    fake_images = self.generator_model(noise)
                    #Create fake output by passing fake images through the discriminator
                    fake_output = self.discriminator_model(fake_images)
                    #Create real output by passing the real images through discriminator
                    real_output = self.discriminator_model(real_images_batch)
                    #Calculate fake loss by using the loss function on the fake output and comparing it to 0s
                    fake_output_discriminator_labels = [0 for x in range(batch_size)]
                    fake_output_discriminator_labels = tf.reshape(fake_output_discriminator_labels, shape = (batch_size, 1))
                    fake_loss = self.loss_function(fake_output_discriminator_labels, fake_output)
                    #Calculate the real loss by using the loss function on the real output and comparing it to 1s
                    real_output_discriminator_labels = [1 for x in range(batch_size)]
                    real_output_discriminator_labels = tf.reshape(real_output_discriminator_labels, shape=(batch_size, 1))
                    real_loss = self.loss_function(real_output_discriminator_labels, real_output)
                    #Add up the 2 losses to calculate total loss
                    total_discriminator_loss = fake_loss + real_loss
                    #total_discriminator_loss = tf.reduce_sum(total_discriminator_loss)/ (batch_size*2) DO NOT REDUCE SUM. IT TAKES AN AVERAGE OF ALL THE REAL AND FAKE LOSSES
                    pass
                #Calculate the gradients between the loss and trainable weights of the discriminator
                gradients = discriminator_tape.gradient(total_discriminator_loss, self.discriminator_model.trainable_variables)
                #Apply those gradients to the trainable weights of the discriminator using the discriminator's optimizer
                self.discriminator_optimizer.apply_gradients(zip(gradients, self.discriminator_model.trainable_variables))
                with tf.GradientTape() as generator_tape:
                    #Create noise with the same dimensions again
                    noise = tf.random.normal(shape=(batch_size, codings_size))
                    #Create fake images by passing the noise through the generator
                    fake_images = self.generator_model(noise)
                    #Create fake output by passing the fake images through the discriminator
                    fake_output = self.discriminator_model(fake_images)
                    #Now we use the loss function to calculate how well the generator is able to trick the discriminator
                    fake_output_discriminator_labels = [1 for x in range(batch_size)]
                    fake_output_discriminator_labels = tf.reshape(fake_output_discriminator_labels, shape=(batch_size, 1))
                    #Calculate the loss, but this time compare the fake output to 1s INSTEAD. This is to trick the discriminator
                    generator_loss = self.loss_function(fake_output_discriminator_labels, fake_output)
                    #generator_loss = tf.reduce_sum(generator_loss)/ (batch_size)
                    pass
                #Calculate the gradients between the loss and trainable weights of the generator
                gradients = generator_tape.gradient(generator_loss, self.generator_model.trainable_variables)
                #Apply those gradients to the trainable weights of the generator using the generator's optimizer
                self.generator_optimizer.apply_gradients(zip(gradients, self.generator_model.trainable_variables))
            pass
            end_time = time.time()
            print("Time for epoch {0} is {1:4f}s || Discriminator loss = {2} || Generator loss = {3}".format(epoch, (end_time-start_time), tf.reduce_sum(total_discriminator_loss), tf.reduce_sum(generator_loss)))
            self.save_the_model_checkpoint(epoch_number=epoch)
            test_image_noise = tf.random.normal(shape=(1, codings_size))
            test_image = self.generator_model(test_image_noise)[0]
            Visualize_data.display_single_image(test_image)
            pass

        pass

    def save_the_model_checkpoint(self, epoch_number):
        self.generator_model.save("Saved_models/ModelEpoch {}".format(epoch_number))
        pass




