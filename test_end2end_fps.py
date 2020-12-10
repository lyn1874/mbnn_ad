#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 09:58:03 2019
This script is for testing the end2end experiment for the sum shortcut connection
@author: li
"""
import tensorflow as tf
import models.multi_branch_clean as mb
import time
from data import read_frame_temporal as rft
import shutil
import numpy as np
import os
import argparse
import evaluate as ev
import math
import const
from utils import read_test_index

def get_anomaly_score(args, version, opt="save_score_faster", tds_name="tds_fps/"):
    if "avenue" in args.data_set:
        data_set_temp = "avenue"
    test_index_all, gt = read_test_index(data_set_temp)
    if "save_score" in opt:
        time_tot = []
        for single_test_index in test_index_all:
            print(single_test_index)
            time_use = run_test(args, single_test_index, version, opt=opt)
            time_tot.append(time_use)
        print("FPS--------", 1/np.mean(time_tot))
#         print("FPS2", 15324/np.sum(time_tot))
    else:
        auc_score = run_test(args, test_index_all, version, opt, gt=gt, tds_name=tds_name)
        print("AUC------", np.round(auc_score*100, 2))


def run_test(args, test_index_use, version, opt, gt=None, tds_name="tds_fps"):
    path_for_load_data = args.datadir
    model_mom = args.expdir
    tds_mom = model_mom
    if "single_branch" in args.model_type or "build_baseline" in args.model_type:
        args.shortcut_connection = False
    else:
        args.shortcut_connection = True  
    args.crop_im = False
    args.aug_opt=None
    args.manipulate_latent = "none"
    model_base = model_mom + '%s_%s/' % (args.model_type, args.data_set)
    tds_base = tds_mom + '%s_%s/' % (args.model_type, args.data_set)
    use_str = ["fore_penalty" if "for_bg" in args.model_type else "motion_penalty"][0]
    
    model_dir = model_base + 'gap_%d_%s_%.3f_numbg_%d_version_%d' % (args.single_interval, use_str,
                                                            args.motion_penalty, args.num_bg, version)
    tds_dir = tds_base+tds_name
    tds_dir = tds_dir + "/gap_%d_%s_%.3f_numbg_%d_version_%d" % (args.single_interval, 
                                                               use_str, args.motion_penalty, args.num_bg,
                                                               version)
    
    if "avenue" in args.data_set:
        if "robust" in args.data_set:
            tr_time = ["robust_on_rain" if "robust_on_rain" in data_set else "robust_on_bright_and_rain"][0]
        else:
            tr_time = None
        data_set = "avenue"
    else:
        tr_time = None
    tds_dir_for_auc = tds_dir + "/Rain_%s_Bright_%s/" % (args.rain_type, args.brightness)
    args.data_set = data_set
    if opt is "calc_auc":
        auc_score = ev.get_auc_score_end2end_sum(tds_dir_for_auc,
                                                 test_index_use, gt)
    else:
        tmf = TestMainFunc(args, path_for_load_data, model_dir, tds_dir, test_index_use, opt, tr_time=tr_time)
        if "single_branch" in args.model_type or "build_baseline" in args.model_type:
            time_use = tmf.calc_fps()
        else:
            time_use = tmf.calc_fps_multi_branch()
    if opt is "calc_auc":
        return auc_score
    else:
        return time_use


class TestMainFunc(object):
    def __init__(self, args, path_for_load_data, model_dir, tds_dir, test_index_use, opt, tr_time=None):
        if not os.path.exists(tds_dir):
            os.makedirs(tds_dir)
        im_filename_stat = rft.read_data(args, path_for_load_data, tr_time=None,
                                         test_index_use=test_index_use)
        im_filenames, train_im_interval, imshape, targ_shape, in_shape, out_shape = im_filename_stat
        batch_size = 2
        args.batch_size = batch_size
        print("The batch_size", args.batch_size)
        args.output_dim = targ_shape[-1]
        args.num_frame = args.time_step + 1
        self.im_filenames = im_filenames
        self.temp_shape = [in_shape, out_shape]
        self.targ_shape = targ_shape
        self.imshape = imshape
        self.data_set = args.data_set
        self.model_dir = model_dir
        self.tds_dir = tds_dir
        tds_dir_use = self.tds_dir + "/Rain_%s_Bright_%s/" % (args.rain_type, args.brightness)
        if not os.path.exists(tds_dir_use):
            os.makedirs(tds_dir_use)
        self.tds_dir = tds_dir_use        
        self.test_index_use = test_index_use
        self.path_for_load_data = path_for_load_data
        self.test_im = train_im_interval
        self.model_type = args.model_type
        self.batch_size = args.batch_size
        self.interval = args.single_interval
        self.delta = args.delta
        self.concat = args.concat_option
        self.time_step = args.time_step
        self.num_bg = args.num_bg
        self.num_encoder_block = args.num_encoder_block
        if args.model_type == "build_baseline":
            args.learn_opt = "learn_fore"
        elif args.model_type == "build_baseline_no_bg_subtraction":
            args.learn_opt = "learn_full_no_bg_subtraction"
        else:
            args.learn_opt = "learn_full"
        self.learn_opt = args.learn_opt
        self.opt = opt
        
        self.rain_type = args.rain_type
        self.brightness = args.brightness
    
        self.aug_opt = args.aug_opt
        self.norm_input = args.norm
        self.manipulate_latent = args.manipulate_latent
        self.shortcut_connection = args.shortcut_connection
        self.shortcut_opt = args.shortcut_opt
        self.tr_time = tr_time

    def read_tensor_npy_type(self):
        """This function reads the original frame and apply the augmentation on each of them instead of on each of
        sequence
        1. A placeholder for the original image filename
        2. A different batch size than the model batch size, because I will need to read as much data as possible
        per iterations
        3. Same as before, I will have a placeholder for darker value, rain type.
        4. Then after I read the data, I will apply the read_frame_interval function and read the frames in
        sequence"""
        imh, imw, ch = self.targ_shape
        batch_size_orig = 200
        self.batch_size_orig = batch_size_orig
        im_stat_output = rft.read_tensor(args, self.path_for_load_data, self.data_set, self.targ_shape, self.imshape,
                                         self.temp_shape, "test_for_score", tr_time=self.tr_time,
                                         batch_size=batch_size_orig)
        images_in, x_input, image_init, x_background, x_real_input, darker_value_tf = im_stat_output
        self.x_input_per_frame = tf.squeeze(x_input, axis=1)  # [batch_size, imh, imw, ch]
        x_input_for_model = tf.placeholder(tf.float32, shape=[self.time_step+1, self.batch_size, imh, imw, ch])
        if self.learn_opt is "learn_fore":
            x_background = rft.calc_mean_std_data(self.path_for_load_data, self.data_set, self.tr_time,
                                                  targ_shape=self.targ_shape)
            x_background = tf.reshape(x_background, shape=[1, 1, imh, imw, ch])
        return images_in, x_input_for_model, image_init, x_background

    def give_image_in_npy_mode(self, image_init, image_placeholder, orig_iter):
        im_tot = []
        imh, imw, ch = self.targ_shape
        self.sess.run(image_init.initializer, feed_dict={image_placeholder: np.expand_dims(self.im_filenames, -1)})
        for i in range(orig_iter):
            im_tot.append(self.sess.run(fetches=self.x_input_per_frame))
        im_tot = [v for j in im_tot for v in j]
        im_tot = np.reshape(im_tot, [np.shape(self.im_filenames)[0], imh, imw, ch])
        return im_tot

    def build_graph(self):
        image_placeholder, x_input, image_init, x_background_manually_calculate = self.read_tensor_npy_type()
        mb_model = mb.MultiBranch(args)
        if self.model_type == "single_branch" or "build_baseline" in self.model_type:
            x_for_model_placeholder, z_from_enc, \
                z_motion_place, z_gt_place, z_mse = mb_model.single_branch_fps(self.targ_shape)
        elif self.model_type == "multi_branch_z":
            x_for_model_placeholder, z_from_enc, \
                z_motion_place, z_gt_place, z_mse = mb_model.multi_branch_z_fps(self.targ_shape)
            z_mse = tf.stack(z_mse, axis=0)
        placeholder_tot = [x_for_model_placeholder, z_motion_place, z_gt_place]
        input_tot = [image_placeholder, image_init, x_input]
        var = tf.trainable_variables()
        saver = tf.train.Saver(var)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        v_all = os.listdir(self.model_dir)
        v_all = [v for v in v_all if '.meta' in v]
        v_all = sorted(v_all, key=lambda s: int(s.strip().split('ckpt-')[1].strip().split('.meta')[0]))
        v_all = v_all[-1]
        model_index = int(v_all.strip().split('.meta')[0].strip().split("-")[-1])
        saver.restore(self.sess, os.path.join(self.model_dir, 'model.ckpt-%d' % model_index))
        return input_tot, placeholder_tot, z_from_enc, z_mse

    def calc_fps(self):
        tf.reset_default_graph()
        imh, imw, ch = self.targ_shape
        input_tot, placeholder_tot, z_from_enc, z_mse = self.build_graph()
        image_placeholder, image_init, x_input = input_tot

        #  -----This is for giving the images in npy mode------------------------------#
        orig_iter = int(np.ceil(np.shape(self.im_filenames)[0] / self.batch_size_orig))
        im_tot = self.give_image_in_npy_mode(image_init, image_placeholder, orig_iter)
        im_tot_sequence, _, _ = rft.read_frame_interval(im_tot, self.time_step, "conc_tr", [self.interval],
                                                        self.delta)
        im_tot_sequence = np.array(im_tot_sequence)
        im_tot_sequence = np.transpose(im_tot_sequence, (1, 0, 2, 3, 4))  # [num_frame, num_sequence, imh, imw, ch]
        im_output_sequence = im_tot_sequence[-1:]  # [1, num_sequence, imh, imw, ch]
        #  ----Finish loading the images----------------------------------------------#
        mse_value = np.zeros(np.shape(im_tot_sequence)[1])
        time_tot = []
        iter_visualize = np.shape(im_tot_sequence)[1] // self.delta
        print("There are supposed to be -----", iter_visualize * (self.delta // self.interval))
        x_for_model_placeholder, z_motion_placeholder, z_gt_placeholder = placeholder_tot
        fh, fw, f_ch = z_gt_placeholder.get_shape().as_list()[-3:]
        latent_space_for_saving = np.zeros([self.delta, fh, fw, f_ch])
        #  x_for_model_placeholder: [time_step+1, batch_size, imh, imw, ch]
        #  z_motion_placeholder: [num_frame, batch_size, fh, fw, ch]
        #  z_gt_placeholder: [batch_size, fh, fw, ch]
        #  z_from_enc: [1, (time_step+1) * batch_size, fh, fw, ch]
        data_iter = 0
        for single_iter in range(iter_visualize):
            if single_iter == 0:
                x_input_npy = im_tot_sequence[:, data_iter*self.batch_size:(data_iter+1)*self.batch_size]
                x_input_npy = np.reshape(x_input_npy, [1, self.batch_size*(self.time_step+1), imh, imw, ch])
                data_iter += 1
                latent_space_value = self.sess.run(fetches=z_from_enc, feed_dict={x_for_model_placeholder: x_input_npy})
                latent_space_update = latent_space_value[0]
                print(np.shape(latent_space_update))
                latent_space_to_motion = latent_space_update[:-self.batch_size]
                latent_space_gt_npy = latent_space_update[-self.batch_size:]

                latent_space_for_saving[0:self.batch_size] = latent_space_gt_npy
                latent_space_to_motion_reshape = np.reshape(latent_space_to_motion, [self.time_step, self.batch_size,
                                                                                     fh, fw, f_ch])
                _mse_npy = self.sess.run(fetches=z_mse, feed_dict={z_motion_placeholder: latent_space_to_motion_reshape,
                                                                   z_gt_placeholder: latent_space_gt_npy})
                mse_value[0:self.batch_size] = _mse_npy
                for j in range(self.delta // self.batch_size - 1):
                    x_input_npy_new = im_tot_sequence[-2:, data_iter*self.batch_size:(data_iter+1)*self.batch_size]
                    x_input_npy_new = np.reshape(x_input_npy_new, [1, self.batch_size*2, imh, imw, ch])
                    data_iter += 1
                    latent_space_for_new_data = self.sess.run(fetches=z_from_enc[0],
                                                              feed_dict={x_for_model_placeholder: x_input_npy_new})
                    latent_space_gt_npy = latent_space_for_new_data[-self.batch_size:]
                    latent_space_to_motion = np.concatenate([latent_space_to_motion[self.batch_size:],
                                                             latent_space_for_new_data[:self.batch_size]], axis=0)
                    latent_space_to_motion_reshape = np.reshape(latent_space_to_motion, [self.time_step,
                                                                                         self.batch_size, fh, fw, f_ch])
                    _mse_npy = self.sess.run(fetches=z_mse,
                                             feed_dict={z_motion_placeholder: latent_space_to_motion_reshape,
                                                        z_gt_placeholder: latent_space_gt_npy})
                    mse_value[single_iter*self.delta + (j + 1) * self.batch_size:
                              single_iter*self.delta + (j + 2) * self.batch_size] = _mse_npy
                    latent_space_for_saving[(j + 1) * self.batch_size:(j + 2) * self.batch_size] = latent_space_gt_npy
                latent_space_to_motion_old = latent_space_to_motion
            else:
                time_init = time.time()
                for j in range(self.delta // self.interval):
                    x_output_npy = im_output_sequence[:, data_iter*self.batch_size:(data_iter+1)*self.batch_size]
                    data_iter += 1
                    latent_space_update_gt = self.sess.run(fetches=z_from_enc,
                                                           feed_dict={x_for_model_placeholder: x_output_npy})
                    latent_space_to_motion = np.concatenate([latent_space_to_motion_old[self.batch_size:],
                                                             latent_space_for_saving[:self.batch_size]], axis=0)
                    latent_space_for_saving = np.concatenate([latent_space_for_saving[self.batch_size:],
                                                              latent_space_update_gt[0]], axis=0)
                    latent_space_to_motion_reshape = np.reshape(latent_space_to_motion,
                                                                [self.time_step, self.batch_size, fh, fw, f_ch])
                    _mse_npy = self.sess.run(fetches=z_mse,
                                             feed_dict={z_motion_placeholder: latent_space_to_motion_reshape,
                                                        z_gt_placeholder: latent_space_update_gt[0]})
                    mse_value[single_iter * self.delta + j * self.batch_size:
                              single_iter * self.delta + (j+1)*self.batch_size] = _mse_npy
                    latent_space_to_motion_old = latent_space_to_motion
                time_end = time.time()
                time_tot.append(time_end - time_init)
        row = np.where(mse_value != 0)[0]
        mse_value = mse_value[row]
        mse_value = np.reshape(mse_value, [-1, 1])

        np.save(os.path.join(self.tds_dir, 'pred_score_%s' % (self.test_index_use)), mse_value)

        return np.mean(time_tot) / self.delta
#         return np.sum(time_tot)

    def op_for_multi_branch(self, z_mse, z_motion_placeholder, z_gt_placeholder, latent_space_to_motion_reshape_group,
                            latent_space_gt_npy_group):
        if self.num_bg == 2:
            _mse_ = self.sess.run(fetches=z_mse,
                                  feed_dict={z_motion_placeholder[0]: latent_space_to_motion_reshape_group[0],
                                             z_motion_placeholder[1]: latent_space_to_motion_reshape_group[1],
                                             z_gt_placeholder[0]: latent_space_gt_npy_group[0],
                                             z_gt_placeholder[1]: latent_space_gt_npy_group[1]})
        return _mse_

    def calc_fps_multi_branch(self):
        tf.reset_default_graph()
        imh, imw, ch = self.targ_shape
        input_tot, placeholder_tot, z_from_enc, z_mse = self.build_graph()
        image_placeholder, image_init, x_input = input_tot
        #  -----This is for giving the images in npy mode------------------------------#
        orig_iter = int(np.ceil(np.shape(self.im_filenames)[0] / self.batch_size_orig))
        im_tot = self.give_image_in_npy_mode(image_init, image_placeholder, orig_iter)
        im_tot_sequence, _, _ = rft.read_frame_interval(im_tot, self.time_step, "conc_tr", [self.interval],
                                                        self.delta)
        im_tot_sequence = np.array(im_tot_sequence)
        im_tot_sequence = np.transpose(im_tot_sequence, (1, 0, 2, 3, 4))  # [num_frame, num_sequence, imh, imw, ch]
        im_output_sequence = im_tot_sequence[-1:]  # [1, num_sequence, imh, imw, ch]
        #  ----Finish loading the images----------------------------------------------#
        mse_value = np.zeros([3, np.shape(im_tot_sequence)[1]])
        time_tot = []
        iter_visualize = np.shape(im_tot_sequence)[1] // self.delta
        print("There are supposed to be -----", iter_visualize * (self.delta // self.interval))
        x_for_model_placeholder, z_motion_placeholder, z_gt_placeholder = placeholder_tot
        fh, fw, f_ch = z_gt_placeholder[0].get_shape().as_list()[-3:]
        latent_space_for_saving = [np.zeros([self.delta, fh, fw, f_ch]) for i in range(self.num_bg)]
        data_iter = 0
        for single_iter in range(iter_visualize):
            if single_iter == 0:
                x_input_npy = im_tot_sequence[:, data_iter*self.batch_size:(data_iter+1)*self.batch_size]
                x_input_npy = np.reshape(x_input_npy, [1, self.batch_size*(self.time_step+1), imh, imw, ch])
                data_iter += 1
                latent_space_value_group = self.sess.run(fetches=z_from_enc,
                                                         feed_dict={x_for_model_placeholder: x_input_npy})
                latent_space_update = [v[0] for v in latent_space_value_group]  # [num_bg, num_frame*batch_size]
                latent_space_to_motion_group = [v[:-self.batch_size] for v in latent_space_update]
                latent_space_gt_npy_group = [v[-self.batch_size:] for v in latent_space_update]
                for bg_iter in range(self.num_bg):
                    latent_space_for_saving[bg_iter][0:self.batch_size] = latent_space_gt_npy_group[bg_iter]
                latent_space_to_motion_reshape_group = [np.reshape(v, [self.time_step, self.batch_size,
                                                                       fh, fw, f_ch]) for v in
                                                        latent_space_to_motion_group]
                _mse_ = self.op_for_multi_branch(z_mse, z_motion_placeholder, z_gt_placeholder,
                                                 latent_space_to_motion_reshape_group,
                                                 latent_space_gt_npy_group)
                mse_value[:, 0:self.batch_size] = _mse_
                for j in range(self.delta // self.batch_size - 1):
                    x_input_npy_new = im_tot_sequence[-2:, data_iter*self.batch_size:(data_iter+1)*self.batch_size]
                    x_input_npy_new = np.reshape(x_input_npy_new, [1, self.batch_size*2, imh, imw, ch])
                    data_iter += 1
                    latent_space_for_new_data = self.sess.run(fetches=z_from_enc,
                                                              feed_dict={x_for_model_placeholder: x_input_npy_new})
                    latent_space_for_new_data = [v[0] for v in latent_space_for_new_data]
                    latent_space_gt_npy_group = [v[-self.batch_size:] for v in latent_space_for_new_data]
                    latent_space_to_motion_group = [np.concatenate([latent_space_to_motion_group[bgi][self.batch_size:],
                                                                    latent_space_for_new_data[bgi][:self.batch_size]],
                                                                   axis=0) for bgi in range(self.num_bg)]
                    latent_space_to_motion_reshape_group = [np.reshape(v, [self.time_step, self.batch_size,
                                                                           fh, fw, f_ch])
                                                            for v in latent_space_to_motion_group]
                    _mse_ = self.op_for_multi_branch(z_mse, z_motion_placeholder, z_gt_placeholder,
                                                     latent_space_to_motion_reshape_group,
                                                     latent_space_gt_npy_group)
                    mse_value[:, single_iter*self.delta + (j + 1) * self.batch_size:
                              single_iter*self.delta + (j + 2) * self.batch_size] = _mse_
                    for bg_iter in range(self.num_bg):
                        latent_space_for_saving[bg_iter][(j+1)*self.batch_size:
                                                         (j+2)*self.batch_size] = latent_space_gt_npy_group[bg_iter]
                latent_space_to_motion_old = latent_space_to_motion_group
            else:
                time_init = time.time()
                for j in range(self.delta // self.interval):
                    x_output_npy = im_output_sequence[:, data_iter*self.batch_size:(data_iter+1)*self.batch_size]
                    data_iter += 1
                    latent_space_update_gt = self.sess.run(fetches=z_from_enc,
                                                           feed_dict={x_for_model_placeholder: x_output_npy})
                    for bg_iter in range(self.num_bg):
                        latent_space_to_motion_group[bg_iter] = np.concatenate([latent_space_to_motion_old[bg_iter][self.batch_size:],
                                                                                latent_space_for_saving[bg_iter][:self.batch_size]],
                                                                               axis=0)
                        latent_space_for_saving[bg_iter] = np.concatenate([latent_space_for_saving[bg_iter][self.batch_size:],
                                                                           latent_space_update_gt[bg_iter][0]], axis=0)
                        latent_space_to_motion_reshape_group[bg_iter] = np.reshape(latent_space_to_motion_group[bg_iter],
                                                                                   [self.time_step, self.batch_size,
                                                                                    fh, fw, f_ch])
                    _mse_ = self.op_for_multi_branch(z_mse, z_motion_placeholder, z_gt_placeholder,
                                                     latent_space_to_motion_reshape_group,
                                                     [v[0] for v in latent_space_update_gt])
                    mse_value[:, single_iter * self.delta + j * self.batch_size:
                              single_iter * self.delta + (j+1)*self.batch_size] = _mse_
                    latent_space_to_motion_old = latent_space_to_motion_group
                time_end = time.time()
                time_tot.append(time_end - time_init)
        mse_value = np.transpose(mse_value, (1, 0))  # [num_value, 4]
        row = np.where(mse_value[:, 0] != 0)[0]
        mse_value = mse_value[row]
        np.save(os.path.join(self.tds_dir, 'pred_score_%s' % (self.test_index_use)), mse_value)

        return np.mean(time_tot) / self.delta


if __name__ == '__main__':
    args = const.get_args()
    args = const.give_motion_foreground_penalty(args)
    print("-------------------------------------------------------------------")
    print("------------------argument for current experiment------------------")
    print("-------------------------------------------------------------------")
    for arg in vars(args):
        print(arg, getattr(args, arg))

    get_anomaly_score(args, args.version, opt=args.test_opt)
    get_anomaly_score(args, args.version, opt="calc_auc")

    
    





