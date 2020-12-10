#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 11:14:51 2019
This is the utils script, to put the functions that have been used in 
multiple scripts
@author: li
"""
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt


def save_im(path_group, stat_group, single_epoch):
    for iterr, single_ca in enumerate(stat_group):
        single_ca_path = path_group[iterr]
        for j in range(np.shape(single_ca)[0]):
            im_use = single_ca[j, :]
            print(single_ca_path, np.max(im_use), np.min(im_use))
            shape_use = np.array(np.shape(im_use)[1:])
            canvas = plot_canvas(im_use, shape_use)
            cv2.imwrite(os.path.join(single_ca_path, 'epoch_%d_frame_%d.png' % (single_epoch, j)),
                        canvas.astype('uint8')[:, :, ::-1])


def save_im_for_test(tds_dir, stat_group, name):
    """this function is for saving the image, recons/pred for test time
    Args:
        stat_group = [im, recons, diff] or [im, pred, diff, bg], float32
    """
    nx = len(stat_group)
    ny = 5
    num_im = np.shape(stat_group[0])[0]
    ch = np.shape(stat_group[0])[-1]
    num_ca = num_im // ny
    for single_ca in range(num_ca):
        stat_use = [v[single_ca * ny:(single_ca + 1) * ny] * 255.0 for v in stat_group]
        if ch == 1:
            stat_use = [np.repeat(v, 3, -1) for v in stat_group]
        ca = create_canvas(stat_use, [nx, ny])
        ca = ca.astype('uint8')[:, :, ::-1]
        cv2.imwrite(tds_dir + '/%s_ca_%d.jpg' % (name, single_ca), ca)


def plot_canvas(image, imshape, ny=8):
    if np.shape(image)[0] < ny:
        ny = np.shape(image)[0]
    nx = np.shape(image)[0] // ny
    x_values = np.linspace(-3, 3, nx)
    y_values = np.linspace(-3, 3, ny)
    targ_height, targ_width = imshape[0], imshape[1]
    if np.shape(image)[-1] == 1:
        image = np.repeat(image, 3, -1)
    imshape[-1] = 3
    canvas = np.empty((targ_height * nx, targ_width * ny, 3))
    for i, yi in enumerate(x_values):
        for j, xi in enumerate(y_values):
            canvas[(nx - i - 1) * targ_height:(nx - i) * targ_height, j * targ_width:(j + 1) * targ_width,
            :] = np.reshape(image[i * ny + j], imshape)
    return (canvas * 255.0).astype('uint8')


def give_legend(model_type_group):
    title_group = []
    for single_model in model_type_group:
        if "build_baseline" in single_model and "bg" not in single_model:
            title_group.append("baseline(-FixBG)")
        elif "no_bg_subtraction" in single_model:
            title_group.append("single-branch (full im)")
        elif "single_branch" in single_model:
            title_group.append("multi-background")
        elif "multi_branch_z" in single_model:
            title_group.append("multi-branch")
        elif "multi_branch_p" in single_model:
            title_group.append("multiple-branch-P (- L_BG)")

    return title_group


def create_canvas(image, nx_ny):
    """This function is used to create the canvas for the images
    image: [Num_im, imh, imw, 3]
    nx_ny: the number of row and columns in the canvas
    """
    nx, ny = nx_ny
    x_values = np.linspace(-3, 3, nx)
    y_values = np.linspace(-3, 3, ny)
    targ_height, targ_width, num_ch = np.shape(image[0])[1:]
    canvas = np.empty((targ_height * nx, targ_width * ny, num_ch))
    for i, yi in enumerate(x_values):
        im_use_init = image[i]
        for j, xi in enumerate(y_values):
            im_use_end = im_use_init[j]
            im_use_end[:, 0, :] = [204, 255, 229]
            im_use_end[:, -1, :] = [204, 255, 229]
            im_use_end[0, :, :] = [204, 255, 229]
            im_use_end[-1, :, :] = [204, 255, 229]
            canvas[(nx - i - 1) * targ_height:(nx - i) * targ_height, j * targ_width:(j + 1) * targ_width,
            :] = im_use_end
    return canvas


def crit_multi_reconstruction_pixel(im, delta):
    """this function is for reconstruction
    im: [iterr, num_frame, batch_size, imh, imw, ch] float32 in range(0,1)
    delta: [gap_between_input_and_output, gap_between_output, num_output]
    delta command is as same as the function below
    return: stat_update, num_act_frame"""
    im = np.transpose(im, (1, 0, 2, 3, 4, 5))
    num_frame, iterr, batch_size, imh, imw, ch = np.shape(im)
    im = np.reshape(im, [num_frame, batch_size * iterr, imh * imw * ch])
    stat_update = crit_multi_prediction(im, delta)
    stat_update = np.reshape(stat_update, [-1, imh, imw, ch])
    return stat_update, np.shape(stat_update)[0]


def crit_multi_prediction_pixel(im):
    """this function is for transforming the prediction im
    im: [num_iter, 1, batch_size, imh, imw, ch]
    """
    im = np.squeeze(im, axis=1)
    num_iter, batch_size, imh, imw, ch = np.shape(im)
    im = np.reshape(im, [num_iter * batch_size, imh, imw, ch])
    return im


def crit_multi_prediction(use_stat, delta):
    """this function is for aggregating the score for multiple predictions
    use_stat: [num_prediction, num_im, num_crit_score]
    delta: [gap_between_input_and_out, gap_between_output, num_output]
    Output:
        [actual_num_im, num_crit_score]
    so the input can be actual score, the num_crit_score will be 7
    or the input can be actual image, the num_crit_score: imh*imw*ch
    or the input can be latent space, the num_crit_score: fh*fw*ch
    I need to remember to reshape the actual image and latent space 
    if I pass them into this function
    """
    num_prediction, num_im, num_crit = np.shape(use_stat)
    actual_tot_num = (delta[2] - 1) * delta[1] + num_im
    stat_new_tot = []
    for single_pred in range(num_prediction):
        stat_sub = use_stat[single_pred]
        before = single_pred * delta[1]
        end = actual_tot_num - before - num_im
        if before != 0:
            before_mat = np.zeros([before, num_crit])
            stat_sub = np.concatenate([before_mat, stat_sub], axis=0)
        if end != 0:
            end_mat = np.zeros([end, num_crit])
            stat_sub = np.concatenate([stat_sub, end_mat], axis=0)
        stat_new_tot.append(stat_sub)
    stat_new_tot = np.array(stat_new_tot)
    stat_update = np.zeros([actual_tot_num, num_crit])
    for single_im in range(actual_tot_num):
        multi_stat_for_one_im = stat_new_tot[:, single_im, :]
        not_equal_zero = np.mean(multi_stat_for_one_im, axis=-1) != 0
        left = np.mean(multi_stat_for_one_im[not_equal_zero, :], axis=0)
        stat_update[single_im, :] = left
    return stat_update


def read_test_index(data_set):
    if "avenue" in data_set:
        test_index_use = ["testing_video_%d_" % i for i in range(22)[1:]]
        gt_path = "gt/Avenue_gt.npy"
        gt = np.load(gt_path, allow_pickle=True)
    elif "brugge" in data_set:
        test_index_use = ["Nov_14/Train_0001", "Nov_14/Train_0003"]
        [test_index_use.append("Nov_18/Train_000%d" % v) if v < 10 else test_index_use.append("Nov_18/Train_00%d" % v)
         for v in [2, 3, 7, 8, 9, 10, 13]]
        gt = []

    return test_index_use, gt


def make_dir(new_dir):
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)


def ax_global_get(fig):
    ax_global = fig.add_subplot(111)
    ax_global.spines['top'].set_color('none')
    ax_global.spines['bottom'].set_color('none')
    ax_global.spines['left'].set_color('none')
    ax_global.spines['right'].set_color('none')
    ax_global.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
    ax_global.tick_params(axis='y', pad=0.6)
    ax_global.tick_params(axis='x', pad=1.0)
    return ax_global


