#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 09:59:43 2019
This script is used for visualizing the anomaly detection result on ucsd
@author: li
"""
import os
import numpy as np
import shutil
import models.dynamic_patch_search as dps
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from utils import create_canvas
import data.read_frame_temporal as rft
import cv2


def calc_accumulate_score(score):
    """this function is used to caculate the accumulated score:
        score_new = score[i]-np.min(score[:i-1]))/np.max(score[:i-1]]-np.min(score[:i-1]]))))
    """
    score_new = np.zeros([np.shape(score)[0]])
    for i in range(np.shape(score)[0]):
        if i == 0:
            score_new[i] = score[i]
        else:
            s_min, s_max = np.min(score[:i]), np.max(score[:i])
            print(s_min, s_max)
            s_new = (score[i] - s_min) / (s_max - s_min)
            score_new[i] = s_new
    return score_new


def create_bg_video(bg_ratio, bg_basis, savedir):
    fps_use = 25.0
    shape_use = (np.shape(bg_basis[0])[1] * 2, np.shape(bg_basis[0])[0] * 1)  # [width, height]
    out_name = savedir + "bg_interpolation.avi"
    out = cv2.VideoWriter(filename=out_name,
                          apiPreference=cv2.CAP_FFMPEG,
                          fourcc=cv2.VideoWriter_fourcc(*'DIVX'),
                          fps=fps_use, frameSize=shape_use)
    color_group = ['r', 'g', 'b']
    time_use = ["14:49", "15:49", "16:49", "17:49"]
    for i in range(len(bg_ratio))[::12]:
        aggre_bg = abs(np.sum(bg_basis * np.reshape(bg_ratio[i], [len(bg_basis), 1, 1, 1]), axis=0)).clip(0, 1)
        fig = plt.figure(figsize=(3, 2.4))
        ax = fig.add_subplot(111)
        for j in range(len(bg_basis)):
            ax.plot(np.arange(i+1), bg_ratio[:(i+1), j], color_group[j])
        ax.legend(["bg-%d" % j for j in range(4)[1:]], loc='best', fontsize=8)
        ax.grid(ls=':', alpha=0.5)
        ax.set_xlim((-1, len(bg_ratio)))
        ax.set_ylim((-0.1, 1.1))
        plt.xticks(np.arange(len(bg_ratio)+1)[::3500], time_use)

        plt.savefig(savedir + '/%d.png' % i, dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close('all')
        interp = cv2.imread(savedir + "/%d.png" % i)
        interp = cv2.resize(interp, dsize=(np.shape(bg_basis[0])[1], np.shape(bg_basis[0])[0]))
        frame0 = (aggre_bg * 255.0).astype('uint8')[:,:,::-1]
        frame_final = cv2.hconcat((frame0, interp))
        os.remove(savedir + "/%d.png" % i)
        out.write(frame_final)
    out.release()


def create_video_using_bb(data_set, test_index_use, ano_score, ano_score_z, diff, model_dir,
                          threshold, input_image=None, pred=None, save_video=False):
    """this function is used to show the anomaly detection result with the corresponding bounding box
    1) the diff is used to get the detected bounding box, one thing needs to remember is that now I select
    box based on some criteria. Then I put the box back into the difference frame, and the original frame
    2) At the same, I need to assign the original gt bounding box back to the frame, so the original frame include
    both ground truth bounding box and the detected bounding box
    3) Then I need to plot the anomalous score plot, it include first the threshold. Then the anomalous score,
    then which frames are anomalous.
    4) Finally, the video shows the anomalous score on the top, the bottom is original frame and difference
    This diff is only for the pixel-based experiment
    And the ano_score include two parts. The first part is the ano score from pixel-based, the second part is the ano_score 
    from the latent space based 
    """

    if data_set is "avenue":
        gt_region_path = "/project_scratch/bo/anomaly_data/" + 'Avenue/gt_box/'
        box_index = int(test_index_use.strip().split('_video_')[1].strip().split('_')[0])
        gt_box = np.load(gt_region_path + '%d_label.npy' % box_index)
    if "avenue" in data_set:
        kernel_size = np.ones([45, 45])
        crit = 25 ** 3 * np.percentile(ano_score, 80)  # 95) 95 is for original avenue dataset
    elif "shanghaitech" in data_set:
        kernel_size = np.ones([20, 20])
        crit = 20 ** 2 * 13 * np.percentile(ano_score, 95)
    stride_size = 5
    num_frame = np.shape(ano_score)[0]
    if data_set is "avenue":
        gt_box = gt_box[-num_frame:]
        gt_data = (np.sum(gt_box, axis=-1) != 0).astype('int32')
        bg_name = None
    elif data_set is "shanghaitech":
        bg_name = test_index_use.strip().split('_')[0]
        bg_name = "bg_index_%s" % bg_name
        gt_box = gt_box[-num_frame:]
    elif "ucsd" in data_set:
        bg_name = None
    if len(input_image) > 0:
        if type(input_image[0]) == str:
            imshape = np.shape(cv2.imread(input_image[0]))[:2]
        else:
            imshape = np.shape(input_image[0])[:2]
#     imshape = np.shape(cv2.imre)
#     tr, tt, imshape, targshape = rft.get_video_data(path_for_load_data, data_set).forward(bg_name)
#     tt = [v for v in tt if test_index_use in v]
#     tt = tt[-np.shape(diff)[0]:]
    print("The shape of image path and loaded diff value", np.shape(diff), np.shape(input_image))
    print("There are %d frames with gt %d" % (num_frame, np.shape(gt_data)[0]))
    im = []
    pred_update = []
    diff = [v for v in diff]
    for iterr, single_diff in enumerate(diff):
        single_diff = cv2.resize(single_diff, (imshape[1], imshape[0]))  # prediction error
        binary_mask, pred_box = dps.calculate_score_for_patch(single_diff, kernel_size, stride_size,
                                                              10, crit)
        single_im = cv2.imread(input_image[iterr])[:,:,::-1]/255.0  # I think this should be between 0,1
        single_im = cv2.resize(single_im, dsize=(imshape[1], imshape[0]))[:, :, ::-1]
#         single_pred = pred[iterr]
#         single_pred = cv2.resize(single_pred, dsize=(imshape[1], imshape[0]))[:, :, ::-1]
        if data_set is "avenue":
            single_gt_box = gt_box[iterr]
            gt_bi = np.zeros(np.shape(single_diff), dtype=np.float32)
            if np.sum(single_gt_box) != 0:
                single_gt_box = single_gt_box.astype('int32')
                gt_bi[single_gt_box[0]:single_gt_box[2], single_gt_box[1]:single_gt_box[3]] = 1.0
                single_im[:, :, 1] = single_im[:, :, 1] + gt_bi * 0.5
#                 single_pred[:, :, 1] = single_pred[:, :, 1] + gt_bi * 0.5
        elif data_set is "shanghaitech":
            gt_bi = gt_box[iterr]
            single_im[:, :, 1] = single_im[:, :, 1] + gt_bi * 0.5
        if np.sum(binary_mask) != 0:
            single_im[:, :, 2] = single_im[:, :, 2] + binary_mask * 0.5
#             single_pred[:, :, 2] = single_pred[:, :, 2] + binary_mask * 0.5
        single_im = (single_im * 255.0).clip(0, 255).astype('uint8')
#         single_pred = (single_pred * 255.0).clip(0, 255).astype('uint8')
        single_diff = np.repeat(np.expand_dims(single_diff, axis=-1), 3, -1)
        im.append(single_im)
#         pred_update.append(single_pred)
        diff[iterr] = (single_diff * 255.0).astype('uint8')

    threshold[0] = threshold[0] / np.max(ano_score_z)
    ano_score_z = ano_score_z / np.max(ano_score_z)
    print(threshold)

    if save_video is False:
        num_im = np.shape(im)[0]
        ny = 15
        num_canvas = num_im // ny
        ca_save_dir = model_dir + 'canvas_%s' % test_index_use
        if not os.path.exists(ca_save_dir):
            os.makedirs(ca_save_dir)
        for single_canvas in range(num_canvas):
            im_subset = im[single_canvas * ny:(single_canvas + 1) * ny]
            diff_subset = diff[single_canvas * ny:(single_canvas + 1) * ny]
            stat_tot = [np.array(diff_subset), np.array(im_subset)]
            ca = create_canvas(stat_tot, [2, ny])
            cv2.imwrite(ca_save_dir + '/ca_%d.jpg' % single_canvas, ca.astype('uint8'))

#     pred_p = (ano_score >= threshold[1]).astype('int32')
    pred_z = (ano_score_z >= threshold[0]).astype('int32')
#     accu_p = np.sum([1 for p, g in zip(pred_p, gt_data) if p == 1 and g == 1]) / np.sum([1 for p in gt_data if p == 1])
    accu_z = np.sum([1 for p, g in zip(pred_z, gt_data) if p == 1 and g == 1]) / np.sum([1 for p in gt_data if p == 1])
    figsize = (6, 3)
    ano_score_dir = model_dir + '_ano_score'
    min_value = np.min(np.concatenate([ano_score, ano_score_z], axis=0))
    max_value = np.max(np.concatenate([ano_score, ano_score_z], axis=0))
    if not os.path.exists(ano_score_dir):
        os.makedirs(ano_score_dir)
        for score_iter, single_score in enumerate(ano_score):
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111)
#             ax.plot(np.arange(num_frame), np.repeat(threshold[1], num_frame), 'r', ls=':')
            ax.plot(np.arange(num_frame), np.repeat(threshold[0], num_frame), 'b', ls=':')
            ax.legend(["z-mse-%.2f" % (accu_z * 100)], loc='best', fontsize=9)
            index_higher = score_iter + 1
            ano_gt_label = gt_data[:index_higher]
            start_index = np.where(ano_gt_label == 1)[0]
#             ax.plot(np.arange(score_iter + 1), ano_score[:index_higher], 'r', lw=0.8)
            ax.plot(np.arange(score_iter + 1), ano_score_z[:index_higher], 'b', lw=0.8)

            if np.shape(start_index)[0] != 0:
#                 ax.plot(np.arange(score_iter + 1)[start_index], ano_score[:index_higher][start_index], 'k.',
#                         markersize=1)
                ax.plot(np.arange(score_iter + 1)[start_index], ano_score_z[:index_higher][start_index], 'k.',
                        markersize=1)
            ax.grid(ls=':', alpha=0.5)
            ax.set_xlim((-1, num_frame))
            ax.set_ylim((min_value, max_value))
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            plt.savefig(ano_score_dir + '/%d.png' % score_iter, pad_inches=0, bbox_inches='tight')
            plt.close('all')
    fps_use = 15.0
    shape_use = (np.shape(im)[2] * 2, np.shape(im)[1]) # * 2)  # [width, height]
    out_name = model_dir + '_vis_anomaly_result.avi'
    out = cv2.VideoWriter(filename=out_name,
                          apiPreference=cv2.CAP_FFMPEG,
                          fourcc=cv2.VideoWriter_fourcc(*'DIVX'),
                          fps=fps_use, frameSize=shape_use)

    f1_shape = (np.shape(im)[1], np.shape(im)[2])
    for i in range(np.shape(ano_score)[0]):
        frame1 = cv2.imread(ano_score_dir + '/%d.png' % i)
        frame1 = cv2.resize(frame1, dsize=(f1_shape[1], f1_shape[0]))
#         frame4 = pred_update[i]
        frame2 = im[i]
#         frame3 = diff[i]
        frame_final = cv2.hconcat((frame2, frame1))
#         frame11 = cv2.hconcat((frame1, frame3))
#         frame22 = cv2.hconcat((frame2, frame4))
#         frame_final = cv2.vconcat((frame22, frame11))
        out.write(frame_final)
    out.release()
    shutil.rmtree(ano_score_dir)
