#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 12:19:52 2019
This is the tensorflow version of these loss functions
@author: li
"""
import tensorflow as tf
import numpy as np


def pure_vae_loss_func(recon_x, x, z_mu, z_var, beta):
    """This z_var is the lograthm of the variance
    so the kl divergence term is 
    kl = 0.5*(var_x+mu^2-1-log(var_x))
    kl = 0.5*(tf.exp(z_var)+mu^2-1-z_var)
    """
    # - N E_q0 [ ln p(x|z_k) ] #this loss is for prediction loss: [batch_size, num_im, imh, imw, ch]
    recons_loss = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(recon_x, x), [-1, -2, -3]), [0, 1])  # correct sum
    kl_term = 0.5 * tf.reduce_mean(tf.reduce_sum(tf.exp(z_var) + tf.square(z_mu) - 1 - z_var, (1)))
    tot_loss = recons_loss + kl_term * beta

    return tot_loss, recons_loss, kl_term


def calculate_gradient(gen_frames, gt_frames):
    """this function is used to calculate the gradient for per frame
    frame shape: [batch_size, imh, imw, ch]
    """
    channels = gen_frames.get_shape().as_list()[-1]
    pos = tf.constant(np.identity(channels), dtype=tf.float32)  # 3 x 3
    neg = -1 * pos
    filter_x = tf.expand_dims(tf.stack([neg, pos]), 0)  # [-1, 1]
    filter_y = tf.stack([tf.expand_dims(pos, 0), tf.expand_dims(neg, 0)])  # [[1],[-1]]
    strides = [1, 1, 1, 1]  # stride of (1, 1)
    padding = 'SAME'

    gen_dx = tf.abs(tf.nn.conv2d(gen_frames, filter_x, strides, padding=padding, name='grad_loss_0'))
    gen_dy = tf.abs(tf.nn.conv2d(gen_frames, filter_y, strides, padding=padding, name='grad_loss_1'))
    gt_dx = tf.abs(tf.nn.conv2d(gt_frames, filter_x, strides, padding=padding, name='grad_loss_2'))
    gt_dy = tf.abs(tf.nn.conv2d(gt_frames, filter_y, strides, padding=padding, name='grad_loss_3'))

    grad_diff_x = tf.abs(gt_dx - gen_dx)
    grad_diff_y = tf.abs(gt_dy - gen_dy)

    grad_loss = tf.reduce_mean(tf.reduce_sum(grad_diff_x + grad_diff_y, (-1, -2, -3)))

    return grad_loss


def calculate_cosine_dist(gen_frames, gt_frames):
    """Since the tensorflow implementation require the input to be unit-normalized,
    I first need to normalize the ground truth latent space, and the predicted latent 
    space, then I can pass them to the tf.loss.cosine_dist function
    return:
        [num_frame, batch_size]
    """
    num_frame, batch_size = gen_frames.get_shape().as_list()[:2]
    gen_frames = tf.reshape(gen_frames, [num_frame, batch_size, -1])
    gt_frames = tf.reshape(gt_frames, [num_frame, batch_size, -1])
    gen_frames = tf.divide(gen_frames, tf.sqrt(tf.reduce_sum(tf.square(gen_frames), axis=-1, keep_dims=True)))
    gt_frames = tf.divide(gt_frames, tf.sqrt(tf.reduce_sum(tf.square(gt_frames), axis=-1, keep_dims=True)))
    loss = tf.losses.cosine_distance(gt_frames, gen_frames, axis=-1, reduction="none")
    return loss


def calculate_cosine_similarity(gen_frame, gt_frame):
    """this function is used to calculate the cosine-similarity between the input,
    the input doesn't need to be normalized"""
    num_frame, batch_size, fh, fw, ch = gen_frame.get_shape().as_list()
    gen_frame = tf.reshape(gen_frame, [num_frame, batch_size, -1])
    gt_frame = tf.reshape(gt_frame, [num_frame, batch_size, -1])
    cos_sim = tf.compat.v1.keras.losses.cosine_similarity(axis=-1)(gen_frame, gt_frame)
    return cos_sim


def test_cosine_loss_func():
    from scipy.spatial.distance import cosine
    from sklearn.metrics.pairwise import cosine_similarity
    gen_frame = np.random.random([2, 4, 4, 3])
    gt_frame = np.random.random([2, 4, 4, 3])
    gen_re = np.reshape(gen_frame, [2, -1])
    gt_re = np.reshape(gt_frame, [2, -1])
    cosine_loss = []
    cos_sim = cosine_similarity(gen_re, gt_re)

    for single_gen, single_gt in zip(gen_re, gt_re):
        _single_cosine = cosine(single_gt, single_gen)
        cosine_loss.append(_single_cosine)
    # ---below is the tensorflow part---#
    gen_frame_tf = tf.constant(gen_frame, dtype=tf.float32)
    gt_frame_tf = tf.constant(gt_frame, dtype=tf.float32)
    cos_loss_tf = calculate_cosine_dist(gen_frame_tf, gt_frame_tf)

    with tf.Session() as sess:
        cos_loss_npy_from_tf = sess.run(fetches=cos_loss_tf)

    numpy_cosine_dist_avg = np.mean(cosine_loss)
    numpy_cosine_sim_avg = np.mean(cos_sim)
    print("----numpy cosine dist %.4f--------------" % (numpy_cosine_dist_avg))
    print("----numpy cosine simi %.4f--------------" % (numpy_cosine_sim_avg))
    print("----cos dist + cos sim %.4f-------------" % (numpy_cosine_dist_avg + numpy_cosine_sim_avg))
    print("----tenso cosine dist %.4f--------------" % (np.mean(cos_loss_npy_from_tf)))
    print("---npy ----- tf---")
    [print(v, j) for v, j in zip(cosine_loss, np.mean(cos_loss_npy_from_tf, axis=(-1, -2)))]


def give_pixel_score(p_x_recons, x_recons_gt, p_x_pred, x_pred_gt):
    """this function is for giving the anomalous score
    recons-mse, recons-psnr
    z-mse, z-cos, z-psnr
    pred-mse, pred-psnr
    Args: 
        p_x_recons: [num_frame,batch_size, imh, imw, ch]
        x_recons_gt: [num_frame, batch_size, imh, imw, ch]
        p_x_pred: [1, batch_size, imh, imw, ch]
        x_pred_gt: [1, batch_size, imh, imw, ch]
        z_latent_pred: [1, batch_size, fh, fw, f_ch]
        z_latent_gt: [1, batch_size, fh, fw, f_ch]
    """

    mse_recons_full = tf.reduce_mean(tf.squared_difference(p_x_recons, x_recons_gt),
                                     (-1, -2, -3))  # num_frame, batch_size
    max_recons = tf.reduce_max(p_x_recons, axis=(-1, -2, -3))

    mse_pred_full = tf.reduce_mean(tf.squared_difference(p_x_pred, x_pred_gt), (-1, -2, -3))  # 1, batch_size
    mse_pred_full = tf.squeeze(mse_pred_full, axis=0)
    max_pred = tf.reduce_max(p_x_pred, axis=(-1, -2, -3))
    max_pred = tf.squeeze(max_pred, axis=0)

    recons_stat = [mse_recons_full, max_recons]
    pred_stat = [mse_pred_full, max_pred]
    return recons_stat, pred_stat


def give_latent_score(latent_space_pred, latent_space_gt, bg_ratio):  # [1,batch_size,num_bg,1,1,1]
    mse_latent_group = []
    cos_latent_group = []
    l1_latent_group = []
    for single_z_pred, single_z_gt in zip(latent_space_pred, latent_space_gt):
        mse_feat = tf.reduce_sum(tf.squared_difference(single_z_gt, single_z_pred), axis=(-1, -2, -3))
        cos_feat = calculate_cosine_dist(single_z_pred, single_z_gt)
        l1_feat = tf.reduce_sum(tf.abs(tf.subtract(single_z_gt, single_z_pred)), axis=(-1, -2, -3))
        cos_feat = tf.squeeze(cos_feat, axis=(0, -1))
        mse_feat = tf.squeeze(mse_feat, axis=0)
        l1_feat = tf.squeeze(l1_feat, axis=0)
        mse_latent_group.append(mse_feat)
        cos_latent_group.append(cos_feat)
        l1_latent_group.append(l1_feat)
    if len(latent_space_pred) != 1:
        bg_ratio = tf.squeeze(bg_ratio, axis=(0, 3, 4, 5))
        num_bg = bg_ratio.get_shape().as_list()[1]
        bg_ratio = tf.unstack(bg_ratio, num_bg, axis=1)
        tot_stat = [mse_latent_group, cos_latent_group, l1_latent_group]
        for iterr, single_stat in enumerate(tot_stat):
            simple_sum = tf.stack([single_stat[0], single_stat[1]], axis=-1)
            simple_sum = tf.reduce_sum(simple_sum, axis=-1)
            aggre_sum = [single_stat[i]*bg_ratio[i] for i in range(num_bg)]
            aggre_sum = tf.reduce_sum(aggre_sum, axis=0)  # batch_size, value
            single_stat.append(simple_sum)
            single_stat.append(aggre_sum)
            tot_stat[iterr] = single_stat
    else:
        tot_stat = [mse_latent_group, cos_latent_group, l1_latent_group]

    tot_stat = [v for q in tot_stat for v in q]
    return tot_stat


def test_latent_score_calc():
    batch_size = 3
    l0 = np.ones(shape=[1, batch_size, 5, 5, 1])+0.5
    l1 = np.ones(shape=[1, batch_size, 5, 5, 1])+0.6
    l2 = np.ones(shape=[1, batch_size, 5, 5, 1])+0.2
    l3 = np.ones(shape=[1, batch_size, 5, 5, 1])+0.1
    l_group = [tf.constant(v, dtype=tf.float32) for v in [l0, l1, l2, l3, l2, l3]]
    bg_ratio_0 = np.ones([1, batch_size, 1, 1, 1, 1])*0.4
    bg_ratio_1 = np.ones([1, batch_size, 1, 1, 1, 1])*0.4
    bg_ratio_2 = np.ones([1, batch_size, 1, 1, 1, 1])*0.2
    bg_ratio = np.concatenate([bg_ratio_0, bg_ratio_1, bg_ratio_2], axis=2)
    bg_ratio = tf.constant(bg_ratio, dtype=tf.float32)
    print(bg_ratio)
    tot_stat = give_latent_score([l_group[0], l_group[1], l_group[4]],
                                 [l_group[2], l_group[3], l_group[5]], bg_ratio)
    sess = tf.Session()
    tot_stat_npy = sess.run(fetches=tot_stat)
    return tot_stat_npy


def train_op(tot_loss, lr, var_opt, name):
    """
    When only the discriminator is trained, the learning rate is set to be 0.0008
    When the generator model is also trained, the learning rate is set to be 0.0004
    Since there are batch_normalization layers in the model, we need to use update_op for keeping train and test moving average
    of the batch_norm parameters
    """
    #    optimizer = tf.train.RMSPropOptimizer(learning_rate = lr)
    epsilon = 1e-4  # added on 18th of July
    optimizer = tf.train.AdamOptimizer(learning_rate=lr, epsilon=epsilon, name=name)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        grads = optimizer.compute_gradients(tot_loss, var_list=var_opt)
        print("================================================")
        print("I am printing the non gradient")
        for grad, var in grads:
            if grad is None:
                print("no gradient", grad, var)
        print("================================================")
        opt = optimizer.apply_gradients(grads)
    return opt
