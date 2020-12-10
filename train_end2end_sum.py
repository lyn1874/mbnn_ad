#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 15:09:35 2019
This script is for training the sum-shortcut connection end2end
I need to make the script as simple as possible!
@author: li
"""
import tensorflow as tf
import models.multi_branch_clean as mb
import optimization.loss_tf as loss_tf
from data import read_frame_temporal as rft
import numpy as np
import os
import argparse
import math
from utils import save_im
import shutil
import const


def train_select(args, version):
    args = const.give_motion_foreground_penalty(args)    
    for single_version in version:
        train(args, single_version)

# # ----------------------------------------------------------------------------#
# # -----------Avenue Day And Night---------------------------------------------#
# # ----------------------------------------------------------------------------$
# def train_daml(args, version):
#     data_set = "avenue"
#     model_type = "daml"
#     time_step, delta, num_layer, single_interval = 6, 6, 4, 2
#     args.batch_size = 5
#     motion_penalty = 0.001
#     args.num_encode_layer = 4
#     args.num_decode_layer = 4
#     for single_version in version:
#         train(args, data_set, model_type, single_interval, delta, single_version,
#               motion_penalty)


# def train_single_branch(args, version):
#     motion_penalty = 0.010
#     data_set = "avenue"
#     args.batch_size = 6
#     model_type = "single_branch"
#     fore_penalty = 0.1
#     for single_version in version:
#         train(args, data_set, model_type, single_interval, delta, single_version, motion_penalty, fore_penalty,
#               shortcut_opt=False)


# def train_multi_z(args, version):
#     motion_penalty = 0.010
#     args.batch_size = 4
#     data_set = "avenue"
#     model_type = "multi_branch_z"
#     fore_penalty = 0.4
#     for single_version in version:
#         train(args, data_set, model_type, single_interval, delta, single_version,
#               motion_penalty, fore_penalty, shortcut_opt=False)
        

# # ----------------------------------------------------------------------------#
# # -----------Avenue different amount of rain----------------------------------#
# # ----------------------------------------------------------------------------#

# def train_single_branch_raining(args, version):
#     model_type = "single_branch"
#     shortcut_opt = False
#     data_set = "avenue_robust_on_rain"
#     args.batch_size = 6
#     single_interval = 2
#     delta = 6
#     fore_penalty = 0.4
#     single_motion_penalty = 0.001
#     for single_version in version:
#         train(args, data_set, model_type, single_interval, delta, single_version, single_motion_penalty, fore_penalty,
#               shortcut_opt)
        

# def train_multi_z_raining(args, version):
#     args.batch_size = 4
#     delta = 6
#     data_set = "avenue_robust_on_rain"
#     model_type = "multi_branch_z"
#     single_interval = 2
#     fore_penalty = 0.4
#     single_motion_penalty = 0.001
#     for single_version in version:
#         train(args, data_set, model_type, single_interval, delta, single_version,
#               single_motion_penalty, fore_penalty, shortcut_opt=False)


def train(args, version):
    path_for_load_data = args.datadir
    model_mom = args.expdir    
    model_dir = model_mom + '/%s_%s/gap_%d_motion_penalty_%.3f_numbg_%d_version_%d' % (args.model_type, args.data_set, 
                                                                                       args.single_interval,
                                                                                       args.motion_penalty, 
                                                                                       args.num_bg, version)
    print(model_dir)
    if "single_branch" in args.model_type or "build_baseline" in args.model_type:
        args.shortcut_connection = False
    else:
        args.shortcut_connection = True
    if "avenue" in args.data_set:
        if "robust" in args.data_set:
            args.aug_opt = "none"
            args.darker_type = "none"
            tr_time = ["robust_on_rain" if "robust_on_rain" in args.data_set else "robust_on_bright_and_rain"][0]
            args.data_set = "avenue"
        else:
            tr_time = None
        data_set = "avenue"
    else:
        args.aug_opt = "none"
        args.darker_type = "none"
        tr_time = None
    if args.model_type is "daml":
        tr_time = "robust_on_rain"
        args.aug_opt = "none"
        args.darker_type = "none"
    args.crop_im = False
    ckpt_dir = None
    tmf = TrainMainFunc(args, path_for_load_data, model_dir, ckpt_dir, tr_time=tr_time)
    tmf.build_running()


class TrainMainFunc(object):
    def __init__(self, args, path_for_load_data, model_dir, ckpt_dir, tr_time=None, train_index=None):
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        _, train_im_interval, imshape, targ_shape, in_shape, out_shape = rft.read_data(args, path_for_load_data,
                                                                                       tr_time, test_index_use=None)
        args.output_dim = targ_shape[-1]
        args.num_frame = args.time_step + 1
        train_im_interval = train_im_interval[np.random.choice(np.arange(np.shape(train_im_interval)[0]),
                                                               np.shape(train_im_interval)[0],
                                                               replace=False)]
        self.temp_shape = [in_shape, out_shape]
        self.targ_shape = targ_shape
        self.imshape = imshape
        self.data_set = args.data_set
        self.train_index = train_index
        self.model_dir = model_dir
        self.path_for_load_data = path_for_load_data
        self.ckpt_dir = ckpt_dir
        self.test_im = train_im_interval
        self.model_type = args.model_type
        self.batch_size = args.batch_size
        self.interval = args.single_interval
        self.concat = args.concat_option
        self.time_step = args.time_step
        self.num_encoder_block = args.num_encoder_block
        if args.model_type == "build_baseline" or args.model_type == "daml":
            args.learn_opt = "learn_fore"
        else:
            args.learn_opt = "learn_full"
        self.learn_opt = args.learn_opt
        if "single_branch" in self.model_type or "build_baseline" in self.model_type or self.model_type == "daml":
            self.learn_bg_epoch = 8
        elif "multi_branch" in self.model_type:
            self.learn_bg_epoch = 10
        self.darker_value = args.darker_value
        self.darker_type = args.darker_type
        self.aug_opt = args.aug_opt
        self.fore_penalty = args.fore_penalty  # this is necessary
        self.motion_penalty = args.motion_penalty  # this is necessary
        self.shortcut_connection = args.shortcut_connection
        self.shortcut_opt = args.shortcut_opt  # this is necessary to change
        args.regu_par = 0.001
        # ----this is for the learning rate ----------------#
        if "build_baseline" in self.model_type or self.model_type == "daml":
            lrate_z_init = 0.00001
            lrate_z_decay_step = 15
            lrate_g_decay_step = 10  # 15
            epoch_for_learn_ae_motion = 30
            max_epoch = 50
        elif self.model_type == "single_branch":  # and tr_time is None:  # I will change the decayrate to 0.5
            lrate_z_init = 0.00005  # 0.0001
            lrate_z_step = 20  # version2:20
            lrate_z_decay_step = 5
            lrate_g_step = 25 + self.learn_bg_epoch
            lrate_g_decay_step = 20  # 25
            epoch_for_learn_ae_motion = lrate_g_step
            max_epoch = lrate_g_step + lrate_z_step
        elif "multi_branch" in self.model_type:
            lrate_g_step = 25 + self.learn_bg_epoch + 5
            lrate_g_decay_step = 18
            lrate_z_decay_step = 5
            lrate_z_init = 0.00005  # 0.00001
            epoch_for_learn_ae_motion = lrate_g_step
            max_epoch = lrate_g_step + 20  # 25+self.learn_bg_epoch+lrate_z_step+5

        self.lrate_g_decay_step = lrate_g_decay_step
        self.lrate_g_init = 0.0001
        self.lrate_z_decay_step = lrate_z_decay_step
        self.lrate_z_init = lrate_z_init
        self.max_epoch = max_epoch
        self.epoch_for_learn_ae_motion = epoch_for_learn_ae_motion
        self.tr_time = tr_time
        print("==========================================================")
        print("There are %d epochs in total" % self.max_epoch)
        print("I train the bg block for %d epoch" % self.learn_bg_epoch)
        print("I train the AE and motion model for the next %d epoch" % epoch_for_learn_ae_motion)
        print("The learning rate for the AE and motion is decayed from %.5f every %d epoch" % (self.lrate_g_init,
                                                                                               self.lrate_g_decay_step))
        print("I train the motion model for the last %d epoch" % (self.max_epoch - epoch_for_learn_ae_motion))
        print("The lr for the motion learning is decayed from %.5f every %d epoch" % (self.lrate_z_init,
                                                                                      self.lrate_z_decay_step))
        print("==========================================================")
        print(args)

    def read_tensor(self):
        images_in, x_input, image_init, \
            x_background, x_real_input, _ = rft.read_tensor(args, self.path_for_load_data, self.data_set,
                                                               self.targ_shape, self.imshape, self.temp_shape,
                                                               "train", tr_time=self.tr_time)
        return images_in, x_input, image_init, x_background

    def build_graph(self):
        imh, imw, ch = self.targ_shape
        image_placeholder, x_input, image_init, x_background_manually_calculate = self.read_tensor()
        mb_model = mb.MultiBranch(args)
        short_place = tf.placeholder(tf.bool, name="shortcut_placeholder")
        if self.model_type == "single_branch" or "build_baseline" in self.model_type:
            background, _, \
                im_stat_group, latent_stat_group = mb_model.single_branch_with_sum_shortcut(x_input)
        elif self.model_type == "daml":  # learn_opt needs to be learn_fore
            background, _, \
                im_stat_group, latent_stat_group = mb_model.single_branch_sota(x_input)
        elif self.model_type == "multi_branch_z":
            background, _, \
                im_stat_group, latent_stat_group = mb_model.multi_branch_aggre_z_sum(x_input)
        elif self.model_type == "multi_branch_z_diff_bg_fg":
            background, _, \
                im_stat_group, latent_stat_group = mb_model.multi_branch_aggre_z_diff_bg_fg(x_input)
        elif self.model_type == "multi_branch_p":
            background, _, \
                im_stat_group, latent_stat_group, _ = mb_model.multi_branch_aggre_p_sum(x_input)
        else:
            print("----The required model doesn't exist yet------------")
        if self.learn_opt == "learn_fore":
            background = x_background_manually_calculate
        elif self.learn_opt == "learn_full_no_bg_subtraction":
            background = tf.constant(0.0, shape=[1, self.batch_size, imh, imw, ch])
        # ------------------Below is for the loss function---------------------------#
        p_x_recons_fore, x_recons_gt_fore, p_x_pred_fore, x_pred_gt_fore = im_stat_group
        if self.model_type == "single_branch" or "build_baseline" in self.model_type or self.model_type == "daml":
            z_pred, z_gt = latent_stat_group
            mse_motion = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(z_pred, z_gt), (-1, -2, -3)))
        elif self.model_type == "multi_branch_z" or self.model_type == \
                "multi_branch_p" or self.model_type == "multi_branch_z_diff_bg_fg":
            mse_motion = []
            for i in range(self.num_encoder_block):
                _mse_motion = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(latent_stat_group[i*2],
                                                                                 latent_stat_group[i*2+1]),
                                                           axis=(-1, -2, -3)))
                mse_motion.append(_mse_motion)
            mse_motion = tf.reduce_sum(mse_motion)
        else:
            mse_motion = tf.constant(0.0)

        p_x_recons_full = p_x_recons_fore + background
        x_recons_full = x_recons_gt_fore + background
        mse_foreground = tf.reduce_mean(
            tf.reduce_sum(tf.squared_difference(p_x_recons_fore, x_recons_gt_fore), (-1, -2, -3)))
        mse_penalty_on_foreground = tf.reduce_mean(tf.reduce_sum(tf.abs(p_x_recons_fore), [-1, -2, -3]))
        mse_full_im = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(p_x_recons_full, x_recons_full), (-1, -2, -3)))
        var = tf.trainable_variables()
        print("The trainable background", [v for v in var if 'trainable_bg' in v.name])
        num_para = [int(np.prod(list(v.shape))) for v in var if '_bg_' not in v.name]
        print("The total number of parameters", np.sum(num_para)/1e+6)
        g_lrate_placeholder = tf.placeholder(tf.float32, name="g_lrate")
        if self.learn_opt == "learn_full":
            loss_0 = mse_foreground + self.fore_penalty * mse_penalty_on_foreground
            var_0 = [v for v in var if 'build_motion_model' not in v.name]
            loss_0 = tf.add_n([loss_0, tf.add_n(
                [tf.nn.l2_loss(v) for v in var_0 if 'kernel' in v.name or 'weight' in v.name]) * args.regu_par])
            train_op_0 = loss_tf.train_op(loss_0, g_lrate_placeholder, var_0, name="train_bg")

        loss_1 = mse_full_im + self.motion_penalty * mse_motion
        var_1 = [v for v in var if 'encoder' in v.name or 'decoder' in v.name or 'build_motion_model' in v.name]
        loss_1 = tf.add_n([loss_1, tf.add_n(
            [tf.nn.l2_loss(v) for v in var_1 if 'kernel' in v.name or 'weight' in v.name]) * args.regu_par])
        train_op_1 = loss_tf.train_op(loss_1, g_lrate_placeholder, var_1, name="train_recons")
        if self.learn_opt == "learn_fore" or self.learn_opt == "learn_full_no_bg_subtraction":
            train_op_0 = train_op_1
        z_lrate_placeholder = tf.placeholder(tf.float32, name="z_lrate")
        if "for_bg" in self.model_type:
            train_op_2 = []
        else:
            loss_2 = mse_motion
            var_2 = [v for v in var if 'build_motion_model' in v.name]
            loss_2 = tf.add_n([loss_2, tf.add_n(
                [tf.nn.l2_loss(v) for v in var_2 if 'kernel' in v.name or 'weight' in v.name]) * args.regu_par])
            train_op_2 = loss_tf.train_op(loss_2, z_lrate_placeholder, var_2, name="train_motion")
        saver_set_all = tf.train.Saver(tf.trainable_variables(), max_to_keep=1)
        self.g_lrate_placeholder = g_lrate_placeholder
        self.z_lrate_placeholder = z_lrate_placeholder
        self.short_placeholder = short_place
        input_group = [image_placeholder, image_init]
        loss_group = [mse_foreground, mse_penalty_on_foreground, mse_full_im, mse_motion]
        train_op_group = [train_op_0, train_op_1, train_op_2, saver_set_all]
        stat_group = [background, p_x_recons_full, x_recons_full, p_x_pred_fore + background,
                      x_pred_gt_fore + background]
        return input_group, loss_group, train_op_group, stat_group

    def train_op(self, sess, fetches, g_lrate_npy, z_lrate_npy, num_iter, crit):
        _loss_temp = np.zeros([num_iter, crit])
        for single_iter in range(num_iter):
            _fetche_npy = sess.run(fetches=fetches, feed_dict={self.g_lrate_placeholder: g_lrate_npy,
                                                               self.z_lrate_placeholder: z_lrate_npy,
                                                               self.short_placeholder: self.shortcut_opt})
            _loss_temp[single_iter, :] = _fetche_npy[1]
        return np.mean(_loss_temp, axis=0)

    def val_op(self, sess, fetches, num_iter, crit, path_group, single_epoch):
        _loss_temp = np.zeros([num_iter, crit])
        for single_iter in range(num_iter):
            if single_iter != num_iter - 1:
                _loss_val = sess.run(fetches=fetches[0], feed_dict={self.short_placeholder: self.shortcut_opt})
            else:
                _loss_val, _stat_val = sess.run(fetches=fetches,
                                                feed_dict={self.short_placeholder: self.shortcut_opt})
                
                save_im(path_group, _stat_val, single_epoch)
            _loss_temp[single_iter, :] = _loss_val
        return np.mean(_loss_temp, axis=0)

    def val_op_pure_loss(self, sess, fetches, num_iter, crit):
        _loss_temp = np.zeros([num_iter, crit])
        for single_iter in range(num_iter):
            _loss_val = sess.run(fetches=fetches[0], feed_dict={self.short_placeholder: self.shortcut_opt})
            _loss_temp[single_iter, :] = _loss_val
        return np.mean(_loss_temp, axis=0)

    def build_running(self):
        path_shortname = ["bg", "recons", "recons_gt", "pred", "pred_gt"]
        path_group = []
        for iterr, single_path in enumerate(path_shortname):
            _path = os.path.join(self.model_dir, single_path)
            if not os.path.exists(_path):
                os.makedirs(_path)
            path_group.append(_path)
        with tf.Graph().as_default():
            input_group, loss_group, train_op_group, stat_group = self.build_graph()  # Need to change this
            image_placeholder, image_init = input_group
            saver = train_op_group[-1]
            x_train = self.test_im[:-self.batch_size * 5]
#            x_train = self.test_im[:self.batch_size*2]
            x_val = self.test_im[-self.batch_size * 5:]
            num_tr_iter_per_epoch = np.shape(x_train)[0] // self.batch_size
            num_val_iter_per_epoch = np.shape(x_val)[0] // self.batch_size
            checkpoint_path = self.model_dir + '/model.ckpt'
            num_loss = len(loss_group)
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                sess.run(tf.local_variables_initializer())
                loss_tr_tot = np.zeros([self.max_epoch, num_loss])
                loss_val_tot = []
                print("single epoch", "foreground-mse", "foreground-penalty", "full-recons-mse", "motion-mse")
                for single_epoch in range(self.max_epoch):
                    sess.run(image_init.initializer, feed_dict={image_placeholder: x_train})
                    lrate_g_npy = self.lrate_g_init * math.pow(0.5, math.floor(
                        float(single_epoch) / float(self.lrate_g_decay_step)))
                    lrate_g_npy = np.max([lrate_g_npy, self.lrate_z_init])
                    lrate_z_npy = self.lrate_z_init * math.pow(0.1, math.floor(
                        float(single_epoch - self.epoch_for_learn_ae_motion) / float(self.lrate_z_decay_step)))
                    lrate_z_npy = np.max([lrate_z_npy, 0.00001])
                    if single_epoch <= self.learn_bg_epoch:
                        tr_loss_per_epoch = self.train_op(sess, [train_op_group[0], loss_group[:2]],
                                                          lrate_g_npy, lrate_z_npy, num_tr_iter_per_epoch, num_loss - 2)
                    elif single_epoch < self.epoch_for_learn_ae_motion:
                        tr_loss_per_epoch = self.train_op(sess, [train_op_group[1], loss_group],
                                                          lrate_g_npy, lrate_z_npy, num_tr_iter_per_epoch, num_loss)
                    else:
                        tr_loss_per_epoch = self.train_op(sess, [train_op_group[2], loss_group],
                                                          lrate_g_npy, lrate_z_npy, num_tr_iter_per_epoch, num_loss)

                    loss_tr_tot[single_epoch, :np.shape(tr_loss_per_epoch)[0]] = tr_loss_per_epoch
                    print("Training", ['%.2f' % i for i in loss_tr_tot[single_epoch, :]], 'g lrate %.5f' % lrate_g_npy,
                          'z-lrate %.5f' % lrate_z_npy)
                    if single_epoch % 5 == 0 or single_epoch == self.max_epoch - 1:
                        sess.run(image_init.initializer, feed_dict={image_placeholder: x_val})
                        # val_loss_per_epoch = self.val_op_pure_loss(sess, [loss_group, stat_group],
                        #                                            num_val_iter_per_epoch, num_loss)
                        val_loss_per_epoch = self.val_op(sess, [loss_group, stat_group],
                                                         num_val_iter_per_epoch, num_loss, path_group, single_epoch)
                        print("Validation", single_epoch, ['%.2f' % i for i in val_loss_per_epoch])
                        loss_val_tot.append(val_loss_per_epoch)
                    if single_epoch % 5 == 0 or single_epoch == self.max_epoch - 1:
                        np.save(self.model_dir + '/tr_loss', loss_tr_tot)
                        np.save(self.model_dir + '/val_loss', np.array(loss_val_tot))
                        saver.save(sess, checkpoint_path, global_step=single_epoch)
                        
                        
if __name__ == '__main__':
    args = const.get_args()
    print("-------------------------------------------------------------------")
    print("------------------argument for current experiment------------------")
    print("-------------------------------------------------------------------")
    for arg in vars(args):
        print(arg, getattr(args, arg))
    print("-------------------------------------------------------------------")
    train_select(args, [args.version])
    # Alternatively, if you want to run multiple experiments do: train_select(args, [0, 1, 2, 3])