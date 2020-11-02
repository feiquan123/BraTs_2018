from __future__ import print_function, division
import os

# no GPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# import tensorflow as tf
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction = 0.3
# session = tf.Session(config=config)

from keras.layers import Input,Concatenate,BatchNormalization
from keras.layers.advanced_activations import LeakyReLU,ReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv2DTranspose
from resnet34 import ResNet34
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.activations import *
# from load_data import *
from utils import SEblock, mini_inception, category_label
import random
import time
import matplotlib.pyplot as plt
import cv2

import numpy as np


class GAN():
    def __init__(self):
        self.img_rows = 64
        self.img_cols = 64
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.df=16
        patch = int(self.img_rows / 2 ** 4)
        self.disc_patch = (patch, patch, 1)

        optimizer = Adam(0.0002, beta_1=0.5)

        # Build and compile the discriminator
        base_discriminator = self.build_discriminator()
        self.discriminator = Model(inputs=base_discriminator.inputs, outputs=base_discriminator.outputs)
        self.discriminator.compile(loss='mse',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        # Build the generator

        base_generator = self.build_generator()
        self.generator = Model(inputs=base_generator.inputs, outputs=base_generator.outputs)

        img = Input(shape=self.img_shape)
        label = Input(shape=self.img_shape)
        fake_label = self.generator(img)

        # For the combined model we will only train the generator
        # self.discriminator.trainable = False
        frozen_D = Model(inputs=base_discriminator.inputs, outputs=base_discriminator.outputs)
        frozen_D.trainable = False
        # The discriminator takes generated images as input and determines validity
        validity = frozen_D([fake_label,img])

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model([label,img], [validity,fake_label])
        self.combined.compile(loss=['mse','mae'],loss_weights=[1,100], optimizer=optimizer)

    def build_generator(self):

        base_model, endpoints = ResNet34(include_top=False, input_shape=self.img_shape)

        x = Conv2DTranspose(filters=1024, kernel_size=(4, 4), strides=(2, 2), padding='same')(
            base_model.output)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Concatenate()([x, endpoints['stage_3/tensor']])
        x = SEblock(x, 1024, 16, 'SE1')
        x = mini_inception(x, 'Icp1', 256, 512)

        x = Conv2DTranspose(filters=512, kernel_size=(4, 4), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Concatenate()([x, endpoints['stage_2/tensor']])
        x = SEblock(x, 512, 16, 'SE2')
        x = mini_inception(x, 'Icp2', 128, 256)

        x = Conv2DTranspose(filters=256, kernel_size=(4, 4), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Concatenate()([x, endpoints['stage_1/tensor']])
        x = SEblock(x, 256, 16, 'SE3')
        x = mini_inception(x, 'Icp3', 64, 128)

        x = Conv2DTranspose(filters=64, kernel_size=(4, 4), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Concatenate()([x, endpoints['stage_0/tensor']])
        x = SEblock(x, 64, 8, 'SE4')
        x = mini_inception(x, 'Icp4', 16, 32)

        x = UpSampling2D()(x)
        x = Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=sigmoid)(x)

        model = Model(base_model.input, x)
        model.summary()
        return model

    def build_discriminator(self):

        # base_model, endpoints = ResNet34(include_top=False, input_shape=self.img_shape)
        def d_layer(layer_input, filters, f_size=4, bn=True):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        # Concatenate image and conditioning image by channels to produce input
        combined_imgs = Concatenate(axis=-1)([img_A, img_B])

        d1 = d_layer(combined_imgs, self.df, bn=False)
        d2 = d_layer(d1, self.df * 2)
        d3 = d_layer(d2, self.df * 4)
        d4 = d_layer(d3, self.df * 8)

        validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)

        return Model([img_A, img_B], validity)


    def generate_gt_img(self, label_train):
        x = get_real_data('./data_ADNI/Label', label_train)
        X = []
        for i in range(len(x)):
            img = preprocess(x[i])
            # X = category_label(X, (64, 64), 2)
            X.append(img)
        X = np.expand_dims(np.asarray(X, dtype=np.float32), -1)
        X = X / 255
        return X

    def generate_input_img(self, label_train):
        x = get_real_data('./data_ADNI/Image', label_train)
        X = []
        for i in range(len(x)):
            img = preprocess(x[i])
            X.append(img)
        X = np.expand_dims(np.asarray(X, dtype=np.float32), -1)
        X = X / 255
        return X

    def train(self, epochs, batch_size=128, sample_interval=1):

        # Load the dataset
        label_all = get_list('./labels_ADNI.txt')
        random.shuffle(label_all)
        label_train = label_all[:-len(label_all) // 10]
        self.label_val = label_all[-len(label_all) // 10:]

        # Adversarial ground truths
        valid = np.ones((batch_size,) + self.disc_patch)
        fake = np.zeros((batch_size,) + self.disc_patch)
        with open('./loss_ADNI.txt','w')as f:
            for epoch in range(epochs):

                for step in range(len(label_train) // batch_size):
                    start_time = time.time()
                    mini_batch = label_train[step * batch_size:(step + 1) * batch_size]
                    input_batch = self.generate_input_img(mini_batch)  # RGB

                    label_batch = self.generate_gt_img(mini_batch)  # gt
                    # Generate a batch of new images
                    gen_imgs = self.generator.predict(input_batch)

                    # Train the discriminator
                    d_loss_real = self.discriminator.train_on_batch([label_batch, input_batch], valid)
                    d_loss_fake = self.discriminator.train_on_batch([gen_imgs, input_batch], fake)
                    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                    f.write(str(d_loss)+',')
                    # ---------------------
                    #  Train Generator
                    # ---------------------

                    # Train the generator (to have the discriminator label samples as valid)
                    g_loss = self.combined.train_on_batch([label_batch, input_batch], [valid, label_batch])
                    f.write(str(g_loss)+'\n')
                    end_time = time.time()
                    # Plot the progress
                    print(
                        "[INFO]epoch:%d step:%d [D loss: %f, acc.: %.2f%%] [G loss: %f G_img_loss: %f] batch time: %.5f" % (
                            epoch, step, d_loss[0], 100 * d_loss[1], g_loss[0], g_loss[1], end_time - start_time))

                # If at save interval => save generated image samples
                if epoch % sample_interval == 0:
                    if not os.path.exists('./weights'):
                        os.mkdir('./weights')
                    self.discriminator.save('./weights/checkpoint_disc_ADNI.h5')
                    self.generator.save('./weights/checkpoint_gen_ADNI.h5')
                    self.sample_images(epoch)

    def sample_images(self, epoch):
        val_file=os.listdir('./data_ADNI/Image/0')
        for item in val_file:
            img = np.expand_dims(
                np.expand_dims(np.asarray(cv2.imread('./data_ADNI/Image/0/'+item, cv2.IMREAD_UNCHANGED), np.float32) / 255, -1), 0)

            gen_imgs = self.generator.predict(img)

            # Rescale images 0 - 1
            gen_imgs = gen_imgs * 255
            gen_imgs[gen_imgs > 255] = 255
            gen_imgs[gen_imgs < 0] = 0
            cv2.imwrite('./val_result/%s' % item, np.squeeze(np.asarray(gen_imgs, np.uint8)))

if __name__ == '__main__':
    gan = GAN()
    gan.train(epochs=300, batch_size=128, sample_interval=1)
