#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 09:58:03 2019
This script is for testing the end2end experiment for the sum shortcut connection
@author: li
"""
import tensorflow as tf
import models.multi_branch_clean as mb
import optimization.loss_tf as loss_tf
import cv2
from data import read_frame_temporal as rft
import shutil
import numpy as np
import os
import argparse
import evaluate as ev
import math
import const
import matplotlib.pyplot as plt
import visualize_video as vv
from utils import crit_multi_prediction_pixel, save_im_for_test
from utils import read_test_index
args = const.get_args()

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def get_anomaly_score(args, version, opt="save_score_faster"):
    test_index_all, gt = read_test_index(args.data_set)
    if "save_score" in opt:
        for single_test_index in test_index_all:
            print(single_test_index)
            run_test(args, single_test_index, 
                     version, opt=opt)
        if "robust" in args.data_set:
            for single_test_index in test_index_all:
                run_test(args, single_test_index,
                         version, opt=opt)
        auc_score = []
    else:
        auc_score = run_test(args, test_index_all, version,
                             opt=opt, gt=gt)        
    return auc_score


def run_test(args, test_index_use, version, opt,
             manipulate_latent=None, gt=None):
    path_for_load_data = args.datadir
    model_mom = args.expdir
    tds_mom = model_mom
    if "single_branch" in args.model_type or "build_baseline" in args.model_type:
        args.shortcut_connection = False
    else:
        args.shortcut_connection = True
    args.crop_im = False
    args.aug_opt = "none"
    args.manipulate_latent = manipulate_latent
    args.norm = False
    model_base = model_mom + '%s_%s/' % (args.model_type, args.data_set)
    tds_base = tds_mom + '%s_%s/' % (args.model_type, args.data_set)
    use_str = ["fore_penalty" if "for_bg" in args.model_type else "motion_penalty"][0]
    motion_penalty = [args.fore_penalty if "for_bg" in args.model_type else args.motion_penalty][0]
    
    model_dir = model_base + 'gap_%d_%s_%.3f_numbg_%d_version_%d' % (args.single_interval, use_str,
                                                            args.motion_penalty, args.num_bg, version)
    if "save_score" in opt or "auc" in opt or opt is "save_video" or opt is "save_diff":
        tds_dir = tds_base+'tds/'
    else:
        tds_dir = tds_base+'tds_video/'
    tds_dir = tds_dir + "gap_%d_%s_%.3f_numbg_%d_version_%d" % (args.single_interval, 
                                                               use_str, args.motion_penalty, args.num_bg,
                                                               version)
    if "avenue" in args.data_set:
        if "robust" in args.data_set:
            tr_time = str(args.data_set.split("avenue_")[1])
        else:            
            tr_time = None
        data_set = "avenue"
    else:
        tr_time = None
    tds_dir_for_auc = tds_dir + "/Rain_%s_Bright_%s/" % (args.rain_type, args.brightness)
    if opt == "calc_auc":
        auc_score = ev.get_auc_score_end2end_sum(tds_dir_for_auc,
                                                 test_index_use, gt)
    elif opt == "save_video":
        create_video(path_for_load_data, tds_dir_for_auc, args.data_set, test_index_use, darker_value, rain_type,
                     manipulate_latent)
    else:
        if not os.path.isfile(tds_dir_for_auc + "/pred_score_%s.npy" % test_index_use):
            tmf = TestMainFunc(args, path_for_load_data, model_dir, tds_dir, test_index_use, opt, tr_time=tr_time)
            if opt == "check_recons_pred":
                tmf.check_recons_pred()
            elif opt == "save_score_faster":
                tmf.save_score_faster()
            elif opt == "check_background":
                tmf.check_background_interpolation()
    if opt == "calc_auc":
        return auc_score, tds_dir


class TestMainFunc(object):
    def __init__(self, args, path_for_load_data, model_dir, tds_dir, test_index_use, opt, tr_time=None):
        if not os.path.exists(tds_dir):
            os.makedirs(tds_dir)
        im_filename_stat = rft.read_data(args, path_for_load_data, tr_time=None,
                                         test_index_use=test_index_use)
        im_filenames, train_im_interval, imshape, targ_shape, in_shape, out_shape = im_filename_stat
        print(train_im_interval[0])
        if "save_score" in opt:
            if "project" in path_for_load_data or "tmp" in path_for_load_data:
                if "multi_branch_p" in args.model_type:
                    batch_max = 200
                else:
                    batch_max = 100
            else:
                batch_max = 60
        else:
            batch_max = 30
        factor = [i for i in range(batch_max)[2:] if np.shape(train_im_interval)[0] % i == 0]
        if factor:
            batch_size = factor[-1]
        else:
            batch_size = 1
        if opt is "check_background":
            batch_size = factor[-1]
        args.batch_size = batch_size
        if args.data_set == "brugge":
            test_index_act = "_".join(test_index_use.split("/"))
        else:
            test_index_act = test_index_use
        print("The batch_size", args.batch_size)
        args.output_dim = targ_shape[-1]
        args.num_frame = args.time_step + 1
        self.im_filenames = im_filenames
        self.temp_shape = [in_shape, out_shape]
        self.targ_shape = targ_shape
        self.imshape = imshape
        self.data_set = args.data_set
        self.rain_type = args.rain_type
        self.brightness = args.brightness

        self.model_dir = model_dir
        self.tds_dir = tds_dir
        self.test_index_use = test_index_act
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
        if args.model_type == "build_baseline" or args.model_type is "daml":
            args.learn_opt = "learn_fore"
        elif args.model_type == "build_baseline_no_bg_subtraction":
            args.learn_opt = "learn_full_no_bg_subtraction"
        else:
            args.learn_opt = "learn_full"
        self.learn_opt = args.learn_opt
        self.opt = opt
        self.aug_opt = args.aug_opt
        self.manipulate_latent = args.manipulate_latent

        self.shortcut_connection = args.shortcut_connection
        self.shortcut_opt = args.shortcut_opt
        self.tr_time = tr_time

    def read_tensor(self):
        images_in, x_input, image_init, \
            x_background, x_real_input, darker_value_tf = rft.read_tensor(args, self.path_for_load_data,
                                                                          self.data_set,
                                                                          self.targ_shape, self.imshape,
                                                                          self.temp_shape, "test",
                                                                          tr_time=self.tr_time)
        return images_in, x_input, image_init, x_background

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

    def build_graph(self):
        imh, imw, ch = self.targ_shape
        if self.opt != "save_score_faster" and self.opt != "save_diff":
            image_placeholder, x_input, image_init, x_background_manually_calculate = self.read_tensor()
            print("The manullay loaded background", x_background_manually_calculate)
        else:
            image_placeholder, x_input, image_init, x_background_manually_calculate = self.read_tensor_npy_type()
        shortcut_placeholder = tf.placeholder(tf.bool, name="sum_shortcut")
        mb_model = mb.MultiBranch(args)
        if self.model_type == "single_branch" or "build_baseline" in self.model_type:
            background, background_ratio, \
                im_stat_group, latent_stat_group = mb_model.single_branch_with_sum_shortcut(x_input)
        elif self.model_type == "daml":
            background, background_ratio, \
                im_stat_group, latent_stat_group = mb_model.single_branch_sota(x_input)
        elif self.model_type == "multi_branch_z":
            background, background_ratio, \
                im_stat_group, latent_stat_group = mb_model.multi_branch_aggre_z_sum(x_input, self.manipulate_latent)
        elif self.model_type == "multi_branch_z_diff_bg_fg":
            background, [background_ratio, foreground_ratio], \
                im_stat_group, latent_stat_group = mb_model.multi_branch_aggre_z_diff_bg_fg(x_input,
                                                                                            self.manipulate_latent)
        elif self.model_type == "multi_branch_p":
            background, background_ratio, \
                im_stat_group, latent_stat_group, branch_stat = mb_model.multi_branch_aggre_p_sum(x_input,
                                                                                                  self.manipulate_latent
                                                                                                  )
        if "_p" not in self.model_type:
            branch_stat = []
        else:
            branch_stat = [tf.clip_by_value(v+background, 0.0, 1.0) for v in branch_stat]
        if self.learn_opt == "learn_fore":
            background = x_background_manually_calculate
        elif self.learn_opt == "learn_full_no_bg_subtraction":
            background = tf.constant(0.0, shape=[1, self.batch_size, imh, imw, ch])

        # ------------------Below is for the loss function---------------------------#
        if self.opt != "check_background":
            im_stat_group = [tf.clip_by_value(v+background, 0.0, 1.0) for v in im_stat_group]
            im_stat_group.append(background)
        if "for_bg" not in self.model_type:
            if "single_branch" in self.model_type or "build_baseline" in self.model_type or "daml" in self.model_type:
                z_pred, z_gt = latent_stat_group
            elif "multi_branch" in self.model_type:
                z_pred_group, z_gt_group = [], []
                for i in range(self.num_encoder_block):
                    z_pred_group.append(latent_stat_group[i*2])
                    z_gt_group.append(latent_stat_group[i*2+1])

        if self.opt != "check_background" and self.opt != "save_input_im_en_bg":
            # -------------save the score-----------------#
            # recons-mse, recons-psnr, z-mse, z-cos, z-l1, p-mse, p-psnr
            recons_score, pred_score = loss_tf.give_pixel_score(im_stat_group[0], im_stat_group[1],
                                                                im_stat_group[2], im_stat_group[3])
            if "single_branch" in self.model_type or "build_baseline" in self.model_type or "daml" in self.model_type:
                latent_score = loss_tf.give_latent_score([z_pred], [z_gt], 0)
            elif "multi_branch" in self.model_type:
                if self.model_type != "multi_branch_z_diff_bg_fg":
                    latent_score = loss_tf.give_latent_score(z_pred_group, z_gt_group, background_ratio)
                else:
                    latent_score = loss_tf.give_latent_score(z_pred_group, z_gt_group, foreground_ratio)
            if self.data_set == "brugge":
                latent_score = [tf.reduce_mean(v, axis=0) for v in latent_score]
            [latent_score.append(v) for v in pred_score]
            print("----recons score-----")
            [print(v) for v in recons_score]
            print("---pred score-----")
            [print(v) for v in latent_score]
        var = tf.trainable_variables()
        if "build_baseline" not in self.model_type and "daml" not in self.model_type:
            background_basic = [v for v in var if 'trainable_bg_tensor' in v.name]
            self.background_basic = background_basic[0]
            print("The learned background basis is ", self.background_basic)
        saver = tf.train.Saver(var)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        v_all = os.listdir(self.model_dir)
        v_all = [v for v in v_all if '.meta' in v]
        v_all = sorted(v_all, key=lambda s: int(s.strip().split('ckpt-')[1].strip().split('.meta')[0]))
        v_all = v_all[-1]
        model_index = int(v_all.strip().split('.meta')[0].strip().split("-")[-1])
        print("Restore model ckpt", self.model_dir + "/model.ckpt-%d" % model_index)
        saver.restore(self.sess, os.path.join(self.model_dir, 'model.ckpt-%d' % model_index))
        print("Successfully restored the model")
        if "build_baseline" not in self.model_type and "daml" not in self.model_type:
            self.background_ratio = tf.squeeze(background_ratio, axis=(-1, -2, -3))  # [1, batch_size, 2]
        else:
            self.background_ratio = background_ratio
        if self.model_type == "multi_branch_z_diff_bg_fg":
            self.foreground_ratio = tf.squeeze(foreground_ratio, axis=(-1, -2, -3))
        input_group = [image_placeholder, shortcut_placeholder, image_init, x_input]
        if self.opt == "check_background":
            return input_group, [], [], []
        else:
            score_group = [recons_score, latent_score]
            return input_group, im_stat_group, score_group, latent_stat_group, branch_stat

    def check_background_interpolation(self):
        """This function is used to check the bg ratio interpolation.
        The background"""
        tf.reset_default_graph()
        input_group, im_stat_group, _, _ = self.build_graph()
        image_placeholder, shortcut_placeholder, image_init, _ = input_group
        bg_ratio_group = np.zeros([len(self.test_im), self.num_bg])
        bg_ratio_tf = tf.squeeze(self.background_ratio, axis=0)
        num_iter = len(self.test_im) // self.batch_size
        print("There are %d images with %d batch_size" % (len(self.test_im), self.batch_size))
        for i in range(num_iter):
            im_use = self.test_im[self.batch_size*i:(i+1)*self.batch_size]
            self.sess.run(image_init.initializer, feed_dict={image_placeholder: im_use})
            if i == 0:
                _bg_ratio, _bg_basis = self.sess.run(fetches=[bg_ratio_tf, self.background_basic])
            else:
                _bg_ratio = self.sess.run(fetches=bg_ratio_tf)
            bg_ratio_group[i*self.batch_size:(i+1)*self.batch_size, :] = _bg_ratio
        print(np.shape(bg_ratio_group), np.shape(_bg_basis))
        np.save(self.tds_dir+"/%s_bg_interpolation_group" % self.test_index_use, [bg_ratio_group, _bg_basis])
        
    def check_recons_pred(self):
        tf.reset_default_graph()
        input_group, im_stat_group, score_group, _, _ = self.build_graph()
        im_stat_group = im_stat_group[2:]
        num_iter = np.shape(self.test_im)[0] // self.batch_size
        print(len(self.test_im), self.batch_size, num_iter, num_iter * self.batch_size)
        image_placeholder, shortcut_placeholder, image_init, _ = input_group
        pred_tot, pred_gt_tot = [], []
        background_tot = []
        bg_ratio_tot = np.zeros([num_iter * self.batch_size, self.num_bg])
        tot_group = [pred_tot, pred_gt_tot, background_tot]
        tds_dir_use = self.tds_dir + "/Rain_%s_Bright_%s/" % (self.rain_type, self.brightness)
        if not os.path.exists(tds_dir_use):
            os.makedirs(tds_dir_use)
        self.sess.run(image_init.initializer, feed_dict={image_placeholder: self.test_im})
        for single_iterr in range(num_iter):
            if "build_baseline" not in self.model_type and "daml" not in self.model_type:
                _stat_, _bg_ratio_npy = self.sess.run(fetches=[im_stat_group, self.background_ratio])
                bg_ratio_tot[single_iterr * self.batch_size:(single_iterr + 1) * self.batch_size, :] = _bg_ratio_npy[0]
            else:
                _stat_ = self.sess.run(fetches=im_stat_group)
            for single_tot, single_stat in zip(tot_group, _stat_):
                single_tot.append(single_stat)
        tot_group = [np.array(v) for v in tot_group]
        pred_diff = (tot_group[0] - tot_group[1]) ** 2
        pred_stat = [tot_group[0], tot_group[1], pred_diff, tot_group[-1]]
        [print(np.shape(v)) for v in pred_stat]
        pred_stat = [crit_multi_prediction_pixel(v) for v in pred_stat]
        save_im_for_test(self.tds_dir, pred_stat,
                         'pred_%s_%s' % (self.test_index_use,  self.manipulate_latent))
        np.save(tds_dir_use + '/bg_ratio_%s' % (self.test_index_use), bg_ratio_tot)
        np.save(tds_dir_use + "/prediction_difference_%s" % self.test_index_use, pred_diff)
        print("--------------------The average prediction error--%.2f" % np.mean(np.sum(pred_diff, (-1, -2, -3))))

    def give_image_in_npy_mode(self, image_init, image_placeholder, orig_iter):
        im_tot = []
        imh, imw, ch = self.targ_shape
        self.sess.run(image_init.initializer, feed_dict={image_placeholder: np.expand_dims(self.im_filenames, -1)})
        for i in range(orig_iter):
            im_tot.append(self.sess.run(fetches=self.x_input_per_frame))
        im_tot = [v for j in im_tot for v in j]
        im_tot = np.reshape(im_tot, [np.shape(self.im_filenames)[0], imh, imw, ch])
        return im_tot

    def save_score_faster(self):
        tf.reset_default_graph()
        input_group, _, score_group, _, _ = self.build_graph()
        imh, imw, ch = self.targ_shape
        orig_iter = int(np.ceil(np.shape(self.im_filenames)[0] / self.batch_size_orig))
        image_placeholder, shortcut_placeholder, image_init, x_input = input_group
        num_frame = np.shape(self.test_im)[0]
        num_iter_for_score = num_frame // self.batch_size
        im_tot = self.give_image_in_npy_mode(image_init, image_placeholder, orig_iter)
        im_tot_sequence, _, _ = rft.read_frame_interval(im_tot, self.time_step, "conc_tr", [self.interval],
                                                        self.delta)
        im_tot_sequence = np.array(im_tot_sequence)
        im_tot_sequence = np.transpose(im_tot_sequence, (1, 0, 2, 3, 4))
        pred_score_tot = np.zeros([num_frame, len(score_group[1])])
        tds_dir_use = self.tds_dir + "/Rain_%s_Bright_%s/" % (self.rain_type, self.brightness)
        if not os.path.exists(tds_dir_use):
            os.makedirs(tds_dir_use)
        for single_iter in range(num_iter_for_score):
            sub = im_tot_sequence[:, single_iter*self.batch_size: (single_iter+1)*self.batch_size]
            _pred_score = self.sess.run(fetches=score_group[1], feed_dict={x_input: sub})
            for j, single_pred_score in enumerate(_pred_score):
                pred_score_tot[single_iter * self.batch_size:(single_iter + 1) * self.batch_size, j] = single_pred_score
        np.save(os.path.join(tds_dir_use, 'pred_score_%s' % (self.test_index_use)), pred_score_tot)
        
        
def get_video():
    path_for_load_data = "/project_scratch/bo/anomaly_data/"
    tds_dir = "/project/bo/exp_data/single_branch_avenue/tds/gap_2_motion_penalty_0.010_numbg_2_version_0/"
    data_set = "avenue"
    test_index_use = "testing_video_21_"
    rain_type = "heavy"
    bright = 8
    create_video(path_for_load_data, tds_dir, data_set, test_index_use, rain_type, bright)
        
        
def create_video(path_for_load_data, tds_dir, data_set, test_index_use, rain_type, bright):
    tds_dir = tds_dir + "Rain_%s_Bright_%d/" % (rain_type, bright)
    diff = np.load(tds_dir + "prediction_difference_%s.npy" % test_index_use)
    diff = np.sum(diff, axis=-1)
    a, _, b, imh, imw = np.shape(diff)
    diff = np.reshape(diff, [a * b, imh, imw])
    threshold_use = np.load(tds_dir + "opt_threshold.npy")
    ano_score_full = np.load(tds_dir + "pred_score_%s.npy" % test_index_use)
    use_index = [0]
    ano_score_z = ano_score_full[:, use_index[0]]
    ano_score = ano_score_full[:, -2]
    
    data_dir = path_for_load_data + "Avenue/frames/%s_testing/bright_%.2f/" % (rain_type, bright/10.0)
    images_all = sorted([v for v in os.listdir(data_dir) if test_index_use in v])[-len(diff):]
    images_all = [data_dir + v for v in images_all]
    print(np.shape(images_all))
    threshold_use = [threshold_use[v] for v in use_index]
    video_save_folder_group = tds_dir.strip().split('tds/')
    if video_save_folder_group[1][-1] is '/':
        video_save_folder_group[1] = video_save_folder_group[1][:-1]
    video_save_folder = video_save_folder_group[0] + 'tds/' + video_save_folder_group[1] + '_video'
    if not os.path.exists(video_save_folder):
        os.makedirs(video_save_folder)

    vv.create_video_using_bb(data_set, test_index_use, ano_score, ano_score_z, diff, video_save_folder + "/%s" % test_index_use,
                             threshold_use, images_all, save_video=True)



def get_score_from_multi_branch(tot_stat, num_bg):
    """this function is used to extract the best score from the multi-branch stat
    Args:
        tot_stat: [num_darkness, num_stat]
        num_bg: the number of encoders
    """
    stat = tot_stat
    num_stat = np.shape(stat)[1] // (num_bg + 2)
    left = np.shape(stat)[1] % (num_bg + 2)
    if left != 0:
        tot_num_stat = num_stat + 2
    else:
        tot_num_stat = num_stat
    stat_new = np.zeros([np.shape(stat)[0], tot_num_stat])
    for iterr in range(num_stat):
        sub = np.max(stat[:, iterr * (num_bg + 2):(iterr + 1) * (num_bg + 2)], axis=1)
        stat_new[:, iterr] = sub
    if left != 0:
        stat_new[:, -2:] = stat[:, -2:]
    return stat_new


if __name__ == '__main__':
    args = const.get_args()
    args = const.give_motion_foreground_penalty(args)
    print("-------------------------------------------------------------------")
    print("------------------argument for current experiment------------------")
    print("-------------------------------------------------------------------")
    for arg in vars(args):
        print(arg, getattr(args, arg))

    if args.test_opt != "save_score_faster":
        run_test(args, args.test_index_use, args.version, args.test_opt, None, None)
    else:
        get_anomaly_score(args, args.version, opt=args.test_opt) #"save_score_faster")
        get_anomaly_score(args, args.version, opt="calc_auc")

        
    
    

# def print_confidence_interval(data_set, single_model, version_group, motion_penalty, aug_opt="add_dark",
#                               rain_type="heavy", num_bg=None, shared="gpu_users", home=True):
#     dark_value = [np.linspace(0.2, 1.0, 9) if aug_opt is "add_dark" else np.linspace(0.2, 1.0, 5)][0]
#     [avg, conf], _ = get_confidence_interval(data_set, 2, 6, single_model, version_group, motion_penalty,
#                                              dark_value, aug_opt, rain_type, num_bg, shared, home)
#     print("-------------------------------------------------")
#     [print("bright %.2f auc %.2f conf %.2f" % (dark_value[i],
#                                                avg[i][0]*100,
#                                                conf[i][0]*100)) for i in range(np.shape(dark_value)[0])]


# def get_confidence_interval(data_set, single_interval, delta, single_model, version_group, motion_penalty,
#                             dark_value, aug_opt="add_dark", rain_type="heavy", num_bg=None,
#                             shared="gpu_project", home=True):
#     """This function is used to get the confidence interval for each model
#     Args:
#         data_set: "avenue"
#         single_interval: 2
#         delta: 6
#         single_model: "single_branch", "build_baseline", "multi_branch_z", "multi_branch_p"
#         version_group: [number model]
#         motion_penalty: 0.01
#         dark_value: the number of darkenss
#         aug_opt: str, either "add_dark" or "add_rain"
#         rain_type: str, either "original", "heavy", "torrential"
#         num_bg: None
#         shared: "gpu_project", "gpu_users"
#         home: True
#     """
#     if num_bg:
#         args.num_bg = num_bg
#     num_model = np.shape(version_group)[0]
#     num_dark = np.shape(dark_value)[0]
#     auc_score_tot = np.zeros([num_model, num_dark, 5])
#     auc_score_multi_branch = []
#     for iterr, single_version in enumerate(version_group):
#         _single_auc_score = get_anomaly_score_multiple_brightness(data_set, single_interval, delta, single_model,
#                                                                   single_version, "calc_auc", motion_penalty,
#                                                                   dark_value, aug_opt=aug_opt, rain_type=rain_type,
#                                                                   shared=shared, home=home, return_element=True)
#         if "multi_branch" in single_model:
#             auc_score_multi_branch.append(_single_auc_score)
#             _single_auc_score = get_score_from_multi_branch(_single_auc_score, args.num_bg)
#         # print(single_version, _single_auc_score[:, 0])
#         auc_score_tot[iterr] = _single_auc_score
#     avg = np.mean(auc_score_tot, axis=0)
#     conf = 1.96*np.std(auc_score_tot, axis=0)/np.sqrt(num_model)
#     if "multi_branch" in single_model:
#         auc_score_multi_branch = np.array(auc_score_multi_branch)
#         score_multi_branch_avg = np.mean(auc_score_multi_branch, axis=0)
#         score_multi_branch_std = 1.96*np.std(auc_score_multi_branch, axis=0)/np.sqrt(num_model)
#     if "multi_branch" in single_model:
#         return [avg, conf], [score_multi_branch_avg, score_multi_branch_std]
#     else:
#         return [avg, conf], [0, 0]


# def get_anomaly_score_multiple_brightness(data_set, single_interval, delta, model_type, version, opt, motion_penalty,
#                                           dark_value=np.linspace(0.2, 1.4, 7), aug_opt="add_dark", rain_type="heavy",
#                                           num_bg=2, return_element=False):
#     if "single_branch" in model_type or "build_baseline" in model_type:
#         num_crit = 5
#     elif "multi_branch_z" in model_type:
#         num_crit = 3*(args.num_bg + 2) + 2
#     elif "multi_branch_p" in model_type:
#         num_crit = 3*4 + 2
#     auc_score_tot = np.zeros([np.shape(dark_value)[0], num_crit])
#     for iterr, single_value in enumerate(dark_value):
#         auc_score_tds = get_anomaly_score(data_set, single_interval, delta, single_value, model_type, version,
#                                           motion_penalty, aug_opt, rain_type=rain_type, opt=opt,
#                                           shared=shared, home=home)
#         if opt is "calc_auc":
#             auc_score_tot[iterr, :] = auc_score_tds[0]
#     if return_element is False:
#         if opt is "calc_auc":
#             if "multi_branch" in model_type:
#                 auc_score_tot = get_score_from_multi_branch(auc_score_tot, args.num_bg)
#             print("===========The anomaly detection accuracy============")
#             for iterr, single_perf in enumerate(auc_score_tot):
#                 print(dark_value[iterr], np.round(single_perf * 100, 2))
#     else:
#         return auc_score_tot