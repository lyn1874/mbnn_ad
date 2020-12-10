#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 5 10:52:11 2019
This script is for the multi-branch work
The idea is to:
    1. Use a small network to learn the ratio for each free background parameters
    2. Subtract the learned background from the original input frames
    3. Use multiple encoders to learn the frames that have been subtracted backgrounds
    4. Use a small network on the extracted latent space to determine the ratio for each of the
    reconstructed foreground
    5. Train the model end2end
@author: li
"""
import tensorflow as tf
import numpy as np


class MotionModel(object):
    def __init__(self, num_frame, ch):
        iterr = int(np.ceil(np.log2(num_frame)))
        if iterr >= 4:
            iterr = 4
        latent_3d_layer = []
        latent_bn_layer = []
        for i in range(iterr):
            latent_3d_layer.append(tf.keras.layers.Conv3D(filters=ch, kernel_size=(2, 3, 3),
                                                          strides=(2, 1, 1), padding='same',
                                                          name="latent_%d" % i))
            latent_bn_layer.append(tf.keras.layers.BatchNormalization(name="latent_%d_bn" % i))
        self.latent_3d_layer = latent_3d_layer
        self.latent_bn_layer = latent_bn_layer
        self.num_iterr_for_motion = iterr

    def forward(self, latent_space, ind=0):
        if isinstance(latent_space, list) is True:
            latent_space_array = tf.concat([latent_space], axis=0)
        else:
            latent_space_array = latent_space
        latent_space_array = tf.transpose(latent_space_array, perm=(1, 0, 2, 3, 4))
        with tf.variable_scope('build_conv3d_motion_%d' % ind):
            for i in range(self.num_iterr_for_motion):
                latent_space_array = self.latent_3d_layer[i](latent_space_array)
                latent_space_array = self.latent_bn_layer[i](latent_space_array)
                if i != self.num_iterr_for_motion - 1:
                    latent_space_array = tf.keras.layers.LeakyReLU()(latent_space_array)
                else:
                    latent_space_array = tf.nn.tanh(latent_space_array)
#                print("the latent space at step", i, latent_space_array)
            left_dim = latent_space_array.get_shape().as_list()[1]
            if left_dim != 1:
                latent_space_array = tf.keras.layers.AveragePooling3D(pool_size=(left_dim, 1, 1))(latent_space_array)
        latent_space_array = tf.squeeze(latent_space_array, axis=1)
        latent_space_array = tf.expand_dims(latent_space_array, axis=0)
        return latent_space_array


def create_block(output_dim, conv_block_index, encoder_index, enc_or_dec):
    conv0 = tf.keras.layers.Conv2D(filters=output_dim, kernel_size=3, strides=1,
                                   padding='same', name='%s_block_%d_conv_0_%d' % (enc_or_dec, conv_block_index,
                                                                                   encoder_index))
    bn0 = tf.keras.layers.BatchNormalization(name="%s_block_%d_batchnorm_0_%d" % (enc_or_dec,
                                                                                  conv_block_index, encoder_index))
    conv1 = tf.keras.layers.Conv2D(filters=output_dim, kernel_size=3, strides=1,
                                   padding='same', name='%s_block_%d_conv_1_%d' % (enc_or_dec,
                                                                                   conv_block_index, encoder_index))
    bn1 = tf.keras.layers.BatchNormalization(name="%s_block_%d_batchnorm_1_%d" % (enc_or_dec,
                                                                                  conv_block_index, encoder_index))
    return conv0, bn0, conv1, bn1


def deconv_layer(output_dim, pool_size, block_index, conv_index):
    layer = tf.keras.layers.Conv2DTranspose(filters=output_dim, kernel_size=pool_size,
                                            strides=pool_size, padding='same',
                                            name='dec_block_%d_deconv_convtranspose_%d' % (block_index,
                                                                                           conv_index))
    return layer


def create_decoder_block(output_dim, pool_size, conv_block_index, decoder_index):
    convtanspose_layer = deconv_layer(output_dim, pool_size, conv_block_index, decoder_index)
    conv0 = tf.keras.layers.Conv2D(filters=output_dim, kernel_size=3, strides=1,
                                   padding='same', name='dec_block_%d_deconv_conv_0_%d' % (conv_block_index,
                                                                                           decoder_index))
    bn0 = tf.keras.layers.BatchNormalization(name="dec_block_%d_bn_0_%d" % (conv_block_index, decoder_index))
    conv1 = tf.keras.layers.Conv2D(filters=output_dim, kernel_size=3, strides=1,
                                   padding='same', name='dec_block_%d_deconv_conv_1_%d' % (conv_block_index,
                                                                                           decoder_index))
    bn1 = tf.keras.layers.BatchNormalization(name="dec_block_%d_bn_1_%d" % (conv_block_index, decoder_index))
    return convtanspose_layer, conv0, bn0, conv1, bn1


def maxpool_layer(single_x, pool_size, block_index):
    single_x, single_x_max_pool_index = tf.nn.max_pool_with_argmax(single_x,
                                                                   ksize=(1, pool_size,
                                                                          pool_size, 1),
                                                                   strides=(1, pool_size,
                                                                            pool_size, 1),
                                                                   padding='SAME',
                                                                   name="block_%d_maxpool" % block_index)
    return single_x, single_x_max_pool_index


def get_encoder_conv_block(num_encoder_layer, feature_root, max_dim, ind):
    enc_conv1, enc_conv2, enc_batch_norm1, enc_batch_norm2 = [], [], [], []
    for i in range(num_encoder_layer):
        output_dim = feature_root * (2 ** i)
        if output_dim >= max_dim:
            output_dim = max_dim
        conv0, batchnorm0, conv1, batchnorm1 = create_block(output_dim, i, ind, "enc")
        for single_list, single_content in zip([enc_conv1, enc_conv2, enc_batch_norm1, enc_batch_norm2],
                                               [conv0, conv1, batchnorm0, batchnorm1]):
            single_list.append(single_content)
    return enc_conv1, enc_conv2, enc_batch_norm1, enc_batch_norm2


def get_decoder_conv_block(data_set, num_decoder_layer, feature_root, output_dim, ind):
    dec_conv1, dec_conv2, dec_batchnorm1, dec_batchnorm2, dec_conv2dtranspose = [], [], [], [], []
    act_num_decoder_layer = num_decoder_layer
    for i in range(act_num_decoder_layer):
        pool_size_use = [2 if "antwerpen" not in data_set and "davis" not in data_set
                         else 2 if i != act_num_decoder_layer - 1
                         else 4][0]
        out_dim = 2 ** (act_num_decoder_layer - i) * feature_root // 2
        deconv_convtranspose, de_conv0, de_bn0, de_conv1, de_bn1 = create_decoder_block(out_dim, pool_size_use,
                                                                                        i, ind)
        for single_list, single_content in zip([dec_conv1, dec_conv2, dec_batchnorm1, dec_batchnorm2],
                                               [de_conv0, de_conv1, de_bn0, de_bn1]):
            single_list.append(single_content)
        dec_conv2dtranspose.append(deconv_convtranspose)
    out_layer = tf.keras.layers.Conv2D(filters=output_dim, kernel_size=3, strides=1,
                                       padding='same', name='final_output_layer_%d' % ind)
    return dec_conv1, dec_conv2, dec_batchnorm1, dec_batchnorm2, dec_conv2dtranspose, out_layer


class UNETshortcut(object):
    def __init__(self, args):
        """This function is used to create an autoencoder with sum shortcut connection"""
        self.num_encoder_layer = args.num_encode_layer
        self.num_decoder_layer = args.num_decode_layer
        self.shortcut_connection = args.shortcut_connection
        # -----------------this is for the encoder part --------------------#
        self.feature_root = 64
        self.output_dim = args.output_dim
        self.data_set = args.data_set

    def build_common_encoder(self, x, ind=0):
        """this function is for build the common encoder
        args: x: [num_frame, batch_size, imh, imw, ch]
        """
        max_dim = 512
        enc_conv1, enc_conv2, \
            enc_batch_norm1, enc_batch_norm2 = get_encoder_conv_block(self.num_encoder_layer,
                                                                      self.feature_root,
                                                                      max_dim, ind)
        num_frame = x.get_shape().as_list()[0]
        with tf.variable_scope('build_sum_encoder_%d' % ind):
            latent_space, feature_maps_all = [], []
            for i in range(num_frame):
                single_x, feature_maps_single_frame = x[i], []
                for j in range(self.num_encoder_layer):
                    pool_size_use = [2 if "antwerpen" not in self.data_set and "davis" not in self.data_set
                                     else 4 if j == 0 else 2][0]
                    single_x = enc_conv1[j](single_x)
                    single_x = enc_batch_norm1[j](single_x)
                    single_x = tf.keras.layers.LeakyReLU()(single_x)
                    single_x = enc_conv2[j](single_x)
                    single_x = enc_batch_norm2[j](single_x)
                    if j != self.num_encoder_layer - 1:
                        single_x = tf.keras.layers.LeakyReLU()(single_x)
                    else:
                        single_x = tf.nn.tanh(single_x)
                    feature_maps_single_frame.append(single_x)
                    single_x, single_x_max_pool_index = maxpool_layer(single_x, pool_size_use,
                                                                      block_index=j)
                latent_space.append(single_x)
                feature_maps_all.append(feature_maps_single_frame)
        return latent_space, feature_maps_all

    def build_common_decoder(self, latent_space, feature_maps, shortcut, ind):
        dec_conv1, dec_conv2, \
            dec_batchnorm1, dec_batchnorm2, \
            dec_conv2dtranspose, out_layer = get_decoder_conv_block(self.data_set, self.num_decoder_layer,
                                                                    self.feature_root, self.output_dim,
                                                                    ind)
        with tf.variable_scope("build_sum_decoder_%d" % ind):
            output_tot = []
            for single_latent, single_feat in zip(latent_space, feature_maps):
                for i in range(self.num_decoder_layer):
                    single_latent = dec_conv2dtranspose[i](single_latent)
                    single_latent = tf.cond(shortcut, lambda: single_latent+single_feat[self.num_decoder_layer-1-i],
                                            lambda: single_latent)
                    single_latent = dec_conv1[i](single_latent)
                    single_latent = dec_batchnorm1[i](single_latent)
                    single_latent = tf.keras.layers.LeakyReLU()(single_latent)
                    single_latent = dec_conv2[i](single_latent)
                    single_latent = dec_batchnorm2[i](single_latent)
                    single_latent = tf.keras.layers.LeakyReLU()(single_latent)
                pred = out_layer(single_latent)
                output_tot.append(pred)
            output_tot = tf.concat([output_tot], axis=0)
        return output_tot


class UNET(object):
    def __init__(self, args):
        """this function is to create an autoencoder that is similar to the unet architecture
        but then I probably need to add the batchnormalization layer and leaky relu
        layer
        """
        self.num_encoder_layer = args.num_encode_layer
        self.num_decoder_layer = args.num_decode_layer
        self.shortcut_connection = args.shortcut_connection
        # -----------------this is for the encoder part --------------------#
        self.feature_root = 64
        self.output_dim = args.output_dim
        self.data_set = args.data_set

    def build_common_encoder(self, x, ind=0):
        """this function is for build the common encoder
        args: x: [num_frame, batch_size, imh, imw, ch]
        """
        max_dim = 512
        enc_conv1, enc_conv2, \
            enc_batch_norm1, enc_batch_norm2 = get_encoder_conv_block(self.num_encoder_layer,
                                                                      self.feature_root,
                                                                      max_dim, ind)
        num_frame = x.get_shape().as_list()[0]
        with tf.variable_scope('build_common_encoder_%d' % ind):
            latent_space, feature_maps_all = [], []
            for i in range(num_frame):
                single_x, feature_maps_single_frame = x[i], []
                for j in range(self.num_encoder_layer):
                    pool_size_use = [2 if "antwerpen" not in self.data_set and "davis" not in self.data_set
                                     else 4 if j == 0 else 2][0]
                    single_x = enc_conv1[j](single_x)
                    single_x = enc_batch_norm1[j](single_x)
                    single_x = tf.keras.layers.LeakyReLU()(single_x)
                    single_x = enc_conv2[j](single_x)
                    single_x = enc_batch_norm2[j](single_x)
                    if j != self.num_encoder_layer - 1:
                        single_x = tf.keras.layers.LeakyReLU()(single_x)
                    else:
                        single_x = tf.nn.tanh(single_x)
                    feature_maps_single_frame.append(single_x)
                    if self.shortcut_connection is True:
                        if j != self.num_encoder_layer - 1:
                            single_x, single_x_max_pool_index = maxpool_layer(single_x, pool_size_use,
                                                                              block_index=j)
                    else:
                        single_x, single_x_max_pool_index = maxpool_layer(single_x, pool_size_use, block_index=j)
                latent_space.append(single_x)
                feature_maps_all.append(feature_maps_single_frame)
        return latent_space, feature_maps_all

    def build_sum_decoder(self, latent_space, ind=0):
        """this function is used to build the sum-operation based shortcut connection
        Args:
            latent_space: [num_frame, batch_size, fh, fw, f_ch]
            ind: int, define the index for each decoder
        """
        if self.shortcut_connection is True:
            act_num_decoder_layer = self.num_decoder_layer - 1
        else:
            act_num_decoder_layer = self.num_decoder_layer
        dec_conv1, dec_conv2, \
            dec_batchnorm1, dec_batchnorm2, \
            dec_conv2dtranspose, out_layer = get_decoder_conv_block(self.data_set, act_num_decoder_layer,
                                                                    self.feature_root, self.output_dim,
                                                                    ind)
        with tf.variable_scope("build_common_decoder_%d" % ind):
            output_tot = []
            for single_latent in latent_space:
                for i in range(act_num_decoder_layer):
                    single_latent = dec_conv2dtranspose[i](single_latent)
                    single_latent = dec_conv1[i](single_latent)
                    single_latent = dec_batchnorm1[i](single_latent)
                    single_latent = tf.keras.layers.LeakyReLU()(single_latent)
                    single_latent = dec_conv2[i](single_latent)
                    single_latent = dec_batchnorm2[i](single_latent)
                    single_latent = tf.keras.layers.LeakyReLU()(single_latent)
                pred = out_layer(single_latent)
                output_tot.append(pred)
            output_tot = tf.concat([output_tot], axis=0)
        return output_tot


class SumDecoder(object):
    def __init__(self, args):
        """This function is for building the decoder and use the same decoder parameter for different latent
        code"""
        self.num_encoder_layer = args.num_encode_layer
        self.num_decoder_layer = args.num_decode_layer
        self.shortcut_connection = args.shortcut_connection
        # -----------------this is for the encoder part --------------------#
        self.feature_root = 64
        self.output_dim = args.output_dim
        self.data_set = args.data_set
        if self.shortcut_connection is True:
            act_num_decoder_layer = self.num_decoder_layer - 1
        else:
            act_num_decoder_layer = self.num_decoder_layer
        dec_conv1, dec_conv2, \
            dec_batchnorm1, dec_batchnorm2, \
            dec_conv2dtranspose, out_layer = get_decoder_conv_block(self.data_set,
                                                                    act_num_decoder_layer,
                                                                    self.feature_root,
                                                                    self.output_dim, 0)
        self.dec_conv1 = dec_conv1
        self.dec_conv2 = dec_conv2
        self.dec_batchnorm1 = dec_batchnorm1
        self.dec_batchnorm2 = dec_batchnorm2
        self.dec_conv2dtranspose = dec_conv2dtranspose
        self.out_layer = out_layer
        self.act_num_decoder_layer = act_num_decoder_layer

    def build_same_decoder(self, latent_space, ind):
        with tf.variable_scope("build_common_decoder_%d" % ind):
            output_tot = []
            for single_latent in latent_space:
                for i in range(self.act_num_decoder_layer):
                    single_latent = self.dec_conv2dtranspose[i](single_latent)
                    single_latent = self.dec_conv1[i](single_latent)
                    single_latent = self.dec_batchnorm1[i](single_latent)
                    single_latent = tf.keras.layers.LeakyReLU()(single_latent)
                    single_latent = self.dec_conv2[i](single_latent)
                    single_latent = self.dec_batchnorm2[i](single_latent)
                    single_latent = tf.keras.layers.LeakyReLU()(single_latent)
                pred = self.out_layer(single_latent)
                output_tot.append(pred)
            output_tot = tf.concat([output_tot], axis=0)
        return output_tot


def blur_input(x_input):
    """Blur the input image with tf.nn.conv2d
    Args:
        x_input: [num_input, batch_size, imh, imw, ch]. It's the frame after subtracting
        the background
    Returns:
        x_blur_input
    """
    print("----I am blurring my input------------")
    ks = 3
    num_input, batch_size, imh, imw, ch = x_input.get_shape().as_list()
    x_input = tf.reshape(x_input, [-1, imh, imw, ch])
    x_input = tf.unstack(x_input, ch, axis=-1)
    kernel = tf.constant(1.0/(ks*ks), shape=[ks, ks, 1, 1], dtype=tf.float32)
    x_input_tf = []
    for single_x in x_input:
        single_x = tf.nn.conv2d(tf.expand_dims(single_x, axis=-1), kernel,
                                strides=(1, 1, 1, 1), padding='SAME', name="blur_input")
        x_input_tf.append(tf.squeeze(single_x, axis=-1))
    x_input_tf = tf.stack(x_input_tf)
    x_input_tf = tf.transpose(x_input_tf, (1, 2, 3, 0))
    x_input_tf = tf.reshape(x_input_tf, [num_input, -1, imh, imw, ch])
    return x_input_tf


def stack_frame(a):
    b = tf.transpose(tf.stack(a), (1, 0))
    return tf.nn.softmax(b)


class MultiBranch(UNET):
    def __init__(self, args):
        super(MultiBranch, self).__init__(args)
        self.num_pred_layer_for_bg = args.num_pred_layer_for_bg
        self.num_bg = args.num_bg
        self.num_frame = args.num_frame
        self.time_step = args.time_step
        self.interval = args.single_interval
        self.delta = args.delta
        self.batch_size = args.batch_size
        self.data_set = args.data_set
        self.shortcut_connection = args.shortcut_connection
        self.learn_opt = args.learn_opt
        self.num_encoder_block = args.num_encoder_block
        # ------this is for building up the bg ratio network----------------#
        self.pred_bg_layer = []
        self.pred_bn_layer = []
        base_feature_for_pred = 4
        for i in range(self.num_pred_layer_for_bg):
            feature_size = base_feature_for_pred * 2 ** i
            _conv = tf.keras.layers.Conv2D(filters=feature_size, kernel_size=4, strides=2, padding='same',
                                           name="pred_bg_conv_%d" % i)
            _bn = tf.keras.layers.BatchNormalization(name="pred_bg_bn_%d" % i)
            self.pred_bg_layer.append(_conv)
            self.pred_bn_layer.append(_bn)
        decoder_model = SumDecoder(args)
        self.base_dim_fore_mask = 8
        self.num_block_fore_mask = 3
        self.decoder_model_individual = decoder_model
        self.args = args

    def build_pred_bg(self, x):
        """this function is used to predict the ratio for the background
        x: [num_frame, batch_size, imh, imw, ch]
        """
        pred_dense_layer_0 = tf.keras.layers.Dense(units=512, name="pred_bg_0", activation=tf.nn.leaky_relu)
        pred_dense_layer_1 = tf.keras.layers.Dense(units=128, name="pred_bg_1", activation=tf.nn.leaky_relu)
        pred_dense_layer_out = tf.keras.layers.Dense(units=self.num_bg, name="pred_bg_out", activation=tf.nn.softmax)
        imh, imw, ch = x.get_shape().as_list()[2:]
        bg_init = tf.zeros([self.num_bg, imh, imw, ch], dtype=tf.float32)
        bg_tensor = tf.Variable(initial_value=bg_init, trainable=True, name="trainable_bg_tensor", dtype=tf.float32)
        bg_ratio = []
        with tf.variable_scope('pred_bg_ratio'):
            for i in range(self.num_frame):
                single_frame = x[i]
                for j in range(self.num_pred_layer_for_bg):
                    single_frame = self.pred_bg_layer[j](single_frame)
                    single_frame = self.pred_bn_layer[j](single_frame)
                    single_frame = tf.keras.layers.LeakyReLU()(single_frame)
                single_frame = tf.reshape(single_frame, shape=[self.batch_size, -1])
                single_frame = pred_dense_layer_0(single_frame)
                single_frame = pred_dense_layer_1(single_frame)
                single_frame = pred_dense_layer_out(single_frame)
                bg_ratio.append(single_frame)
        bg_ratio = tf.concat([bg_ratio], axis=0)  # [num_frame, batch_size, num_bg]
        bg_ratio = tf.reshape(bg_ratio, shape=[self.num_frame, self.batch_size, self.num_bg, 1, 1, 1])
        bg_tensor = tf.expand_dims(tf.expand_dims(bg_tensor, axis=0), axis=0)
        bg_aggregate = tf.multiply(bg_ratio, bg_tensor)
        bg_final = tf.reduce_sum(bg_aggregate, axis=2, name="aggregate_bg")  # [num_frame, batch_size, imh, imw, ch]
        bg_final = tf.reduce_mean(bg_final, axis=0, name="avg_input_frame_for_bg",
                                  keep_dims=True)  # [1, batch, imh, imw, ch]
        bg_final = tf.clip_by_value(bg_final, 0.0, 1.0)
        return bg_final, tf.reduce_mean(bg_ratio, axis=0, keep_dims=True)  # [1, batch_size, num_bg,1,1,1]

    def build_pred_bg_for_three_branches(self, x):
        """this function is used to predict the ratio for the background
        x: [num_frame, batch_size, imh, imw, ch]
        """
        pred_dense_layer_0 = tf.keras.layers.Dense(units=512, name="pred_bg_0", activation=tf.nn.leaky_relu)
        pred_dense_layer_1 = tf.keras.layers.Dense(units=128, name="pred_bg_1", activation=tf.nn.leaky_relu)
        pred_dense_layer_out = tf.keras.layers.Dense(units=self.num_bg+self.num_encoder_block,
                                                     name="pred_bg_out")
        imh, imw, ch = x.get_shape().as_list()[2:]
        bg_init = tf.zeros([self.num_bg, imh, imw, ch], dtype=tf.float32)
        bg_tensor = tf.Variable(initial_value=bg_init, trainable=True, name="trainable_bg_tensor", dtype=tf.float32)
        bg_ratio = []
        fg_ratio = []
        with tf.variable_scope('pred_bg_ratio'):
            for i in range(self.num_frame):
                single_frame = x[i]
                for j in range(self.num_pred_layer_for_bg):
                    single_frame = self.pred_bg_layer[j](single_frame)
                    single_frame = self.pred_bn_layer[j](single_frame)
                    single_frame = tf.keras.layers.LeakyReLU()(single_frame)
                single_frame = tf.reshape(single_frame, shape=[self.batch_size, -1])
                single_frame = pred_dense_layer_0(single_frame)
                single_frame = pred_dense_layer_1(single_frame)
                single_frame = pred_dense_layer_out(single_frame)
                single_frame = tf.unstack(single_frame, self.num_bg+self.num_encoder_block, -1)
                _s_bg_ratio = stack_frame(single_frame[:self.num_bg])
                _s_fg_ratio = stack_frame(single_frame[-self.num_encoder_block:])
                bg_ratio.append(_s_bg_ratio)
                fg_ratio.append(_s_fg_ratio)
        bg_ratio = tf.concat([bg_ratio], axis=0)  # [num_frame, batch_size, num_bg]
        bg_ratio = tf.reshape(bg_ratio, shape=[self.num_frame, self.batch_size, self.num_bg, 1, 1, 1])
        fg_ratio = tf.concat([fg_ratio], axis=0)  # [num_frame, batch_size, num_fg]
        fg_ratio = tf.reshape(fg_ratio, shape=[self.num_frame, self.batch_size, self.num_encoder_block, 1, 1, 1])
        bg_tensor = tf.expand_dims(tf.expand_dims(bg_tensor, axis=0), axis=0)
        bg_aggregate = tf.multiply(bg_ratio, bg_tensor)
        bg_final = tf.reduce_sum(bg_aggregate, axis=2, name="aggregate_bg")  # [num_frame, batch_size, imh, imw, ch]
        bg_final = tf.reduce_mean(bg_final, axis=0, name="avg_input_frame_for_bg",
                                  keep_dims=True)  # [1, batch, imh, imw, ch]
        bg_final = tf.clip_by_value(bg_final, 0.0, 1.0)
        fg_ratio = tf.reduce_mean(fg_ratio, axis=0, keep_dims=True)
        return bg_final, tf.reduce_mean(bg_ratio, axis=0, keep_dims=True), fg_ratio  # [1, batch_size, num_bg,1,1,1]

    def build_pred_bg_for_fps(self, x):
        pred_dense_layer_0 = tf.keras.layers.Dense(units=512, name="pred_bg_0", activation=tf.nn.leaky_relu)
        pred_dense_layer_1 = tf.keras.layers.Dense(units=128, name="pred_bg_1", activation=tf.nn.leaky_relu)
        pred_dense_layer_out = tf.keras.layers.Dense(units=self.num_bg, name="pred_bg_out", activation=tf.nn.softmax)
        imh, imw, ch = x.get_shape().as_list()[2:]  # [1, batch_size*num_frame, imh, imw, ch]
        bg_init = tf.zeros([self.num_bg, imh, imw, ch], dtype=tf.float32)
        bg_tensor = tf.Variable(initial_value=bg_init, trainable=True, name="trainable_bg_tensor", dtype=tf.float32)
        bg_ratio = []
        num_frame = x.get_shape().as_list()[0]
        with tf.variable_scope('pred_bg_ratio'):
            for i in range(num_frame):
                single_frame = x[i]
                for j in range(self.num_pred_layer_for_bg):
                    single_frame = self.pred_bg_layer[j](single_frame)
                    single_frame = self.pred_bn_layer[j](single_frame)
                    single_frame = tf.keras.layers.LeakyReLU()(single_frame)
                f_t_h, f_t_w, f_t_ch = single_frame.get_shape().as_list()[1:]
                single_frame = tf.reshape(single_frame, shape=[-1, f_t_h*f_t_w*f_t_ch])
                single_frame = pred_dense_layer_0(single_frame)
                single_frame = pred_dense_layer_1(single_frame)
                single_frame = pred_dense_layer_out(single_frame)
                bg_ratio.append(single_frame)
        bg_ratio = tf.concat([bg_ratio], axis=0)  # [num_frame, batch_size, num_bg]
        bg_ratio = tf.reshape(bg_ratio, shape=[num_frame, -1, self.num_bg, 1, 1, 1])
        bg_tensor = tf.expand_dims(tf.expand_dims(bg_tensor, axis=0), axis=0)
        bg_aggregate = tf.multiply(bg_ratio, bg_tensor)
        bg_final = tf.reduce_sum(bg_aggregate, axis=2, name="aggregate_bg")  # [num_frame, batch_size, imh, imw, ch]
        bg_final = tf.reduce_mean(bg_final, axis=0, name="avg_input_frame_for_bg",
                                  keep_dims=True)  # [1, batch, imh, imw, ch]
        bg_final = tf.clip_by_value(bg_final, 0.0, 1.0)
        return bg_final, tf.reduce_mean(bg_ratio, axis=0, keep_dims=True)  # [1, batch_size, num_bg,1,1,1]

    def build_foreground_mask(self, x):
        """This function builds the foreground mask
        The input x is the x that has subtract the background
        """
        num_input = x.get_shape().as_list()[0]
        fore_rec_mask_layer = []
        fore_rec_mask_bn_layer = []
        for i in range(self.num_block_fore_mask):
            output_dim = self.base_dim_fore_mask * 2 ** i
            fore_rec_mask_layer.append(tf.keras.layers.Conv2D(filters=output_dim, kernel_size=3,
                                                              strides=1, padding='same',
                                                              name="fore_rec_mask_%d" % i))
            fore_rec_mask_bn_layer.append(tf.keras.layers.BatchNormalization(name="fore_rec_mask_bn_%d" % i))
        mask_output_layer = tf.keras.layers.Conv2D(filters=1, kernel_size=3, strides=1, padding='same',
                                                   name="output_mask_layer")
        print("-------------------the beginning of learning foreground mask-----------------")
        with tf.variable_scope('build_fore_rec_mask'):
            mask_tot = []
            for j in range(num_input):
                x_fake_fore = x[j]
                for i in range(self.num_block_fore_mask):
                    x_fake_fore = fore_rec_mask_layer[i](x_fake_fore)
                    x_fake_fore = fore_rec_mask_bn_layer[i](x_fake_fore)
                    x_fake_fore = tf.keras.layers.LeakyReLU()(x_fake_fore)
                x_fake_fore = mask_output_layer(x_fake_fore)
                x_fake_fore = tf.nn.sigmoid(x_fake_fore)
                mask_tot.append(x_fake_fore)
        fake_fore_mask = tf.concat([mask_tot], axis=0)  # [num_input, batch_size, imh, imw, 1]
        fake_fore_mask = fake_fore_mask[1:]
        print("-------------the learned foreground mask", fake_fore_mask)
        print("-------------------the end of learning foreground mask-----------------")
        return fake_fore_mask

    def single_branch_with_sum_shortcut(self, x):
        """if pred is True, then x: [x_input, x_output]
        else:
        x: [x_input]
        if learn_opt is "learn_fore", then I am building the baseline model, so
        I should not learn the background and subtract the background again
        """
        if self.learn_opt == "learn_full":
            background, background_ratio = self.build_pred_bg(x)
            x = x - background
            x = blur_input(x)
        elif self.learn_opt == "learn_fore":
            background = []
            x = x
            background_ratio = []
        elif self.learn_opt == "learn_full_no_bg_subtraction":
            x = x
            background = []
            background_ratio = []
        else:
            print("The required method doesn't exist", self.learn_opt)
        num_input = x.get_shape().as_list()[0]  # [only include input, does not include output]
        with tf.variable_scope('build_encoder'):
            latent_space, feature_map = self.build_common_encoder(x, ind=0)
        with tf.variable_scope('build_motion_model'):
            f_ch = latent_space[0].get_shape().as_list()[-1]
            motion_model_use = MotionModel(num_input - 1, f_ch)
            latent_space_pred = motion_model_use.forward(latent_space[:-1], ind=0)
            latent_space_gt = latent_space[-1:]
        latent_space_to_decoder = latent_space[1:]
        latent_space_to_decoder[-1] = latent_space_pred[0]
        decoder_output = self.build_sum_decoder(latent_space_to_decoder, ind=0)
        x_actual_use = x[1:]
        p_x_recons = decoder_output[:-1]
        x_recons_gt = x_actual_use[:-1]
        p_x_pred = decoder_output[-1:]
        x_pred_gt = x_actual_use[-1:]
        stat = [p_x_recons, x_recons_gt, p_x_pred, x_pred_gt]
        latent_stat = [latent_space_pred, latent_space_gt]
        return background, background_ratio, stat, latent_stat
    
    def single_branch_sota(self, x):
        """This function uses the architecture from the arxiv paper to demonstrate the benefit of removing
        shortcut connection
        The input is the subtracted foreground frames, so I will directly input it
        """
        num_input = x.get_shape().as_list()[0]
        unet_with_sum_shortcut = UNETshortcut(self.args)
        with tf.variable_scope("build_encoder"):
            latent_space, feature_map = unet_with_sum_shortcut.build_common_encoder(x, 0)
        latent_space_to_decoder = latent_space[1:]  # this is right!
        feature_maps_to_decoder = [feature_map[i] for i in range(num_input - 1)]  # this is also right
        with tf.variable_scope('build_motion_model'):
            f_ch = latent_space[0].get_shape().as_list()[-1]
            motion_model_use = MotionModel(num_input - 1, f_ch)
            latent_space_pred = motion_model_use.forward(latent_space[:-1], ind=0)
            latent_space_gt = latent_space[-1:]
        latent_space_to_decoder[-1] = latent_space_pred[0]
        shortcut_tensor = tf.constant(True, dtype=tf.bool, name="shortcut_placeholder")
        with tf.variable_scope("build_decoder"):
            decoder_output = unet_with_sum_shortcut.build_common_decoder(latent_space_to_decoder,
                                                                         feature_maps_to_decoder,
                                                                         shortcut=shortcut_tensor,
                                                                         ind=0)
        x_actual_use = x[1:]
        p_x_recons = decoder_output[:-1]
        x_recons_gt = x_actual_use[:-1]
        p_x_pred = decoder_output[-1:]
        x_pred_gt = x_actual_use[-1:]
        stat = [p_x_recons, x_recons_gt, p_x_pred, x_pred_gt]
        latent_stat = [latent_space_pred, latent_space_gt]
        return [], [], stat, latent_stat

    def single_branch_fps(self, targshape):
        """This function is as same as the previous function, except that I only use
        this function for calculating the fps
        1. the input for the encoder is a placeholder: [1, (time_step+1)*interval, imh, imw, ch]
        2. These input are passed through the encoder to get the latent code
        3. Then the latent code for the motion model is also placeholder,
           which should be [batch_size, time_step, fh, fw, ch], then I get the predicted latent code
           latent_space_gt and latent_space_pred are both placeholder: [batch_size,fh,fw,ch]
        4. After that, I calculate the mse between the latent-code-gt and latent-code-predict.
        """
        imh, imw, ch = targshape
        x_placeholder = tf.placeholder(dtype=tf.float32, shape=[1, None, imh, imw, ch])
        if self.learn_opt == "learn_full":
            background, background_ratio = self.build_pred_bg_for_fps(x_placeholder)
            x = x_placeholder - background
            x = blur_input(x)
        elif self.learn_opt == "learn_fore":
            x = x_placeholder
        print("-----------Input to the model------------", x)
        with tf.variable_scope('build_encoder'):
            latent_space, feature_map = self.build_common_encoder(x, ind=0)  # [1, time_step+1*interval, fh, fw, ch]
        print("----------z from encoder-----------------", latent_space)
        fh, fw, f_ch = latent_space[0].get_shape().as_list()[1:]
        latent_space_for_motion_placeholder = tf.placeholder(tf.float32,
                                                             shape=[self.time_step, self.batch_size, fh, fw, f_ch],
                                                             name="latent_space_for_motion")
        latent_space_gt = tf.placeholder(tf.float32, shape=[self.batch_size, fh, fw, f_ch],
                                         name="gt_latent_space")
        # -----build motion model-------#
        with tf.variable_scope('build_motion_model'):
            f_ch = latent_space[0].get_shape().as_list()[-1]
            motion_model_use = MotionModel(self.time_step, f_ch)
            latent_space_pred = motion_model_use.forward(latent_space_for_motion_placeholder, ind=0)
            latent_space_pred = tf.squeeze(latent_space_pred, axis=0)
        mse_feat = tf.reduce_sum(tf.squared_difference(latent_space_pred, latent_space_gt), axis=(-1, -2, -3))
        print("---------z mse", mse_feat)
        return x_placeholder, latent_space, latent_space_for_motion_placeholder, latent_space_gt, mse_feat
    
    def multi_branch_aggre_p_sum(self, x, manipulate_latent="none"):
        """this function is used for doing both reconstruction and prediction using multiple
        encoder and multiple decoder. Note, the output p_x_recons and p_x_pred is the
        aggregated version"""
        background, background_ratio = self.build_pred_bg(x)
        num_input = x.get_shape().as_list()[0]
        x = x - background
        latent_space_group = []
        for i in range(self.num_encoder_block):
            _latent_space, _ = self.build_common_encoder(x, ind=i)
            latent_space_group.append(_latent_space)
        print("----------Motion Phase---------------------------------------------")
        f_ch = _latent_space[0].get_shape().as_list()[-1]
        latent_space_pred_group = []
        with tf.variable_scope("build_motion_model"):
            motion_model_use = MotionModel(num_input - 1, f_ch)
            for i in range(self.num_encoder_block):
                _latent_space_pred = motion_model_use.forward(latent_space_group[i][:-1], ind=0)
                print("---encoder---%d" % i, _latent_space_pred)
                latent_space_pred_group.append(_latent_space_pred)
        latent_space_gt_group = [latent_space_group[i][-1:] for i in range(self.num_encoder_block)]
        latent_space_to_decoder_group = [latent_space_group[i][1:] for i in range(self.num_encoder_block)]
        for iterr, single_latent_space_to_decoder in enumerate(latent_space_to_decoder_group):
            single_latent_space_to_decoder[-1] = latent_space_pred_group[iterr][0]
        bg_ratio_unstack = tf.unstack(background_ratio, num=self.num_bg,
                                      axis=2)  # [num_input, self.batch_size, 1, 1, 1]
        decoder_output_group, decoder_aggre = [], []
        for i in range(self.num_encoder_block):
            _decoder_output = self.build_sum_decoder(latent_space_to_decoder_group[i], ind=i)
            _decoder_output_with_bg_ratio = tf.multiply(_decoder_output, bg_ratio_unstack[i])
            decoder_output_group.append(_decoder_output)
            decoder_aggre.append(_decoder_output_with_bg_ratio)
        decoder_aggre = tf.reduce_sum(decoder_aggre, axis=0)
        p_x_recons = decoder_aggre[:-1]
        p_x_pred = decoder_aggre[-1:]
        x_recons_gt = x[1:-1]
        x_pred_gt = x[-1:]
        im_stat = [p_x_recons, x_recons_gt, p_x_pred, x_pred_gt]
        latent_space_group = []
        for single_latent_pred, single_latent_gt in zip(latent_space_pred_group, latent_space_gt_group):
            latent_space_group.append(single_latent_pred)
            latent_space_group.append(single_latent_gt)
        branch_stat = decoder_output_group
        return background, background_ratio, im_stat, latent_space_group, branch_stat

    def multi_branch_aggre_z_diff_bg_fg(self, x, manipulate_latent="none"):
        """this function is used for doing both reconstruction and prediction using multiple encode
        but single decoder
        """
        background, background_ratio, foreground_ratio = self.build_pred_bg_for_three_branches(x)
        num_input = x.get_shape().as_list()[0]
        x = x - background
        latent_space_group = []
        for i in range(self.num_encoder_block):
            _latent_space, _ = self.build_common_encoder(x, ind=i)
            latent_space_group.append(_latent_space)
        print("----------Motion Phase---------------------------------------------")
        f_ch = _latent_space[0].get_shape().as_list()[-1]
        latent_space_pred_group = []
        with tf.variable_scope("build_motion_model"):
            motion_model_use = MotionModel(num_input - 1, f_ch)
            for i in range(self.num_encoder_block):
                _latent_space_pred = motion_model_use.forward(latent_space_group[i][:-1], ind=0)
                print("---encoder---%d" % i, _latent_space_pred)
                latent_space_pred_group.append(_latent_space_pred)
        latent_space_gt_group = [latent_space_group[i][-1:] for i in range(self.num_encoder_block)]
        latent_space_to_decoder_group = [latent_space_group[i][1:] for i in range(self.num_encoder_block)]
        for iterr, single_latent_space_to_decoder in enumerate(latent_space_to_decoder_group):
            single_latent_space_to_decoder[-1] = latent_space_pred_group[iterr][0]
        fg_ratio_unstack = tf.unstack(foreground_ratio, num=self.num_encoder_block,
                                      axis=2)
        if manipulate_latent == "first":
            fg_ratio_unstack[0] = tf.constant(0.0, shape=[1, self.batch_size, 1, 1, 1])
        elif manipulate_latent == "second":
            fg_ratio_unstack[1] = tf.constant(0.0, shape=[1, self.batch_size, 1, 1, 1])
        elif manipulate_latent == "third":
            fg_ratio_unstack[2] = tf.constant(0.0, shape=[1, self.batch_size, 1, 1, 1])
        z_aggre_group = []
        for single_latent_space_to_decoder, single_bg_ratio in zip(latent_space_to_decoder_group,
                                                                   fg_ratio_unstack):
            z_aggre_group.append(single_latent_space_to_decoder*single_bg_ratio)
        z_aggre = tf.reduce_sum(z_aggre_group, axis=0)
        z_aggre = tf.unstack(z_aggre, num_input - 1, axis=0)
        decoder_output = self.build_sum_decoder(z_aggre, ind=0)
        p_x_recons = decoder_output[:-1]
        p_x_pred = decoder_output[-1:]
        x_recons_gt = x[1:-1]
        x_pred_gt = x[-1:]
        im_stat = [p_x_recons, x_recons_gt, p_x_pred, x_pred_gt]
        latent_stat = []
        for single_latent_pred, single_latent_gt in zip(latent_space_pred_group, latent_space_gt_group):
            latent_stat.append(single_latent_pred)
            latent_stat.append(single_latent_gt)
        ratio_group = [background_ratio, foreground_ratio]
        return background, ratio_group, im_stat, latent_stat

    def multi_branch_aggre_z_sum(self, x, manipulate_latent="none"):
        """this function is used for doing both reconstruction and prediction using multiple encode
        but single decoder
        """
        background, background_ratio = self.build_pred_bg(x)
        num_input = x.get_shape().as_list()[0]
        x = x - background
        latent_space_group = []
        for i in range(self.num_encoder_block):
            _latent_space, _ = self.build_common_encoder(x, ind=i)
            latent_space_group.append(_latent_space)
        print("----------Motion Phase---------------------------------------------")
        f_ch = _latent_space[0].get_shape().as_list()[-1]
        latent_space_pred_group = []
        with tf.variable_scope("build_motion_model"):
            motion_model_use = MotionModel(num_input - 1, f_ch)
            for i in range(self.num_encoder_block):
                _latent_space_pred = motion_model_use.forward(latent_space_group[i][:-1], ind=0)
                print("---encoder---%d" % i, _latent_space_pred)
                latent_space_pred_group.append(_latent_space_pred)
        latent_space_gt_group = [latent_space_group[i][-1:] for i in range(self.num_encoder_block)]
        latent_space_to_decoder_group = [latent_space_group[i][1:] for i in range(self.num_encoder_block)]
        for iterr, single_latent_space_to_decoder in enumerate(latent_space_to_decoder_group):
            single_latent_space_to_decoder[-1] = latent_space_pred_group[iterr][0]
        bg_ratio_unstack = tf.unstack(background_ratio, num=self.num_bg,
                                      axis=2)  # [num_input, self.batch_size, 1, 1, 1]
        if manipulate_latent == "first":
            bg_ratio_unstack[0] = tf.constant(0.0, shape=[1, self.batch_size, 1, 1, 1])
        elif manipulate_latent == "second":
            bg_ratio_unstack[1] = tf.constant(0.0, shape=[1, self.batch_size, 1, 1, 1])
        elif manipulate_latent == "third":
            bg_ratio_unstack[2] = tf.constant(0.0, shape=[1, self.batch_size, 1, 1, 1])
        z_aggre_group = []
        for single_latent_space_to_decoder, single_bg_ratio in zip(latent_space_to_decoder_group,
                                                                   bg_ratio_unstack):
            z_aggre_group.append(single_latent_space_to_decoder * single_bg_ratio)
        z_aggre = tf.reduce_sum(z_aggre_group, axis=0)
        z_aggre = tf.unstack(z_aggre, num_input - 1, axis=0)
        decoder_output = self.build_sum_decoder(z_aggre, ind=0)
        p_x_recons = decoder_output[:-1]
        p_x_pred = decoder_output[-1:]
        x_recons_gt = x[1:-1]
        x_pred_gt = x[-1:]
        im_stat = [p_x_recons, x_recons_gt, p_x_pred, x_pred_gt]
        latent_stat = []
        for single_latent_pred, single_latent_gt in zip(latent_space_pred_group, latent_space_gt_group):
            latent_stat.append(single_latent_pred)
            latent_stat.append(single_latent_gt)
        return background, background_ratio, im_stat, latent_stat

    def multi_branch_z_fps(self, targshape):
        """This function is as same as the previous function, except that I only use
        this function for calculating the fps
        1. the input for the encoder is a placeholder: [1, (time_step+1)*interval, imh, imw, ch]
        2. These input are passed through the encoder to get the latent code
        3. Then the latent code for the motion model is also placeholder,
           which should be [batch_size, time_step, fh, fw, ch], then I get the predicted latent code
           latent_space_gt and latent_space_pred are both placeholder: [batch_size,fh,fw,ch]
        4. After that, I calculate the mse between the latent-code-gt and latent-code-predict.
        """
        imh, imw, ch = targshape
        x_placeholder = tf.placeholder(dtype=tf.float32, shape=[1, None, imh, imw, ch])
        if self.learn_opt == "learn_full":
            background, background_ratio = self.build_pred_bg_for_fps(x_placeholder)
            x = x_placeholder - background
            x = blur_input(x)
        elif self.learn_opt == "learn_fore":
            x = x_placeholder
        latent_space_group = []
        for i in range(self.num_encoder_block):
            _latent_space, _ = self.build_common_encoder(x, ind=i)
            latent_space_group.append(_latent_space)
        print("----------z from encoder-----------------", _latent_space)
        fh, fw, f_ch = _latent_space[0].get_shape().as_list()[1:]
        z_for_motion_placeholder_group = []
        z_gt_placeholder_group = []
        z_mse_group = []
        with tf.variable_scope("build_motion_model"):
            motion_model_use = MotionModel(self.time_step, f_ch)
            for i in range(self.num_encoder_block):
                z_for_motion_placeholder = tf.placeholder(tf.float32, shape=[self.time_step, self.batch_size,
                                                                             fh, fw, f_ch],
                                                          name="z_for_prediction_enc_%d" % i)
                z_gt_placeholder = tf.placeholder(tf.float32, shape=[self.batch_size, fh, fw, f_ch],
                                                  name="z_gt_enc_%d" % i)
                _latent_space_pred = motion_model_use.forward(z_for_motion_placeholder, ind=0)
                _latent_space_pred = tf.squeeze(_latent_space_pred, axis=0)  # [batch_size, fh, fw, f_ch]
                z_mse = tf.reduce_sum(tf.squared_difference(_latent_space_pred, z_gt_placeholder), axis=(-1, -2, -3))
                print("---encoder---%d" % i, _latent_space_pred, z_for_motion_placeholder, z_gt_placeholder)
                z_for_motion_placeholder_group.append(z_for_motion_placeholder)
                z_gt_placeholder_group.append(z_gt_placeholder)
                z_mse_group.append(z_mse)
        bg_ratio_unstack = tf.unstack(background_ratio, num=self.num_bg,
                                      axis=2)  # [num_input, self.batch_size, 1, 1, 1]
        # z_new = []
        # for i in range(self.num_bg):
        #     _value = z_mse_group[i] * tf.squeeze(bg_ratio_unstack[i], axis=(0, -1, -2, -3))
        #     z_new.append(_value)
        z_mse_group.append(tf.reduce_sum(z_mse_group, axis=0))
        # z_mse_group.append(tf.reduce_sum(z_new, axis=0))
        return x_placeholder, latent_space_group, z_for_motion_placeholder_group, z_gt_placeholder_group, z_mse_group

