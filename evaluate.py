#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 09:13:01 2019
This document is for evaluating the multi-branch model
@author: li
"""
import numpy as np
import os
from sklearn.metrics import roc_curve, auc


def extract_score(score_tot, num_crit):
    score_latent = []
    for iterr, single_score in enumerate(score_tot):
        single_score = np.array(single_score)
        num_stat_per_metric = (np.shape(single_score)[0] - 2) // 3
        stat_sub = np.zeros([num_crit, 5])
        for single_stat_iter in range(3):
            low = single_stat_iter*num_stat_per_metric
            upper = (single_stat_iter+1)*num_stat_per_metric
            stat_select = single_score[:, low:upper]
            stat_select = np.max(stat_select, axis=-1)
            stat_sub[:, single_stat_iter] = stat_select
        stat_sub[:, -2:] = single_score[:, -2:]  # [pred_mse, pred_psnr]
        score_latent.append(stat_sub)
    return np.array(score_latent)


def get_auc_score_end2end_sum(tds_dir, test_index_all, gt, norm_score=False, show=True):
    auc_score_tot = []
    opt_threshold_tot = []
    single_interval = 2
    if 'single_branch' in tds_dir or "build_baseline" in tds_dir or "daml" in tds_dir:
        pred_legend = ["zmse", "zcos", "zl1", "pred-mse", "pred-psnr"]
    else:
        pred_legend = ["z-mse-0", "z-mse-1", "z-mse-sum", "z-mse-ratio",
                       "z-cos-0", "z-cos-1", "z-cos-sum", "z-cos-ratio",
                       "z-l1-0", "z-l1-1", "z-l1-sum", "z-l1-ratio", "pred-mse", "pred-psnr"]

    recons_legend = ["recons-mse", "recons-psnr"]
    legend_space = np.concatenate([recons_legend, pred_legend], axis=0)
    error_stat = np.zeros([np.shape(legend_space)[0]])

    for use_mark in ["pred_score"]:
        ano_score_tot, gt_tot = [], []
        for iterr, single_test_index in enumerate(test_index_all):
            sub_name = "/%s_%s.npy" % (use_mark, single_test_index)
            ano_score_per_test_video, \
                gt_per_test_video, error_stat = main_part_for_calc_ano_score(use_mark, tds_dir,
                                                                             sub_name, error_stat, single_interval,
                                                                             norm_score, gt[iterr])

            ano_score_tot.append(ano_score_per_test_video)
            gt_tot.append(gt_per_test_video)
        auc_score_per_use_mark, opt_threshold_per_use_mark = give_auc_score(ano_score_tot, gt_tot)
        auc_score_tot.append(auc_score_per_use_mark)
        opt_threshold_tot.append(opt_threshold_per_use_mark)
    auc_score = np.concatenate(auc_score_tot)
    opt_threshold = np.concatenate(opt_threshold_tot)
    # error_stat = error_stat / np.shape(test_index_all)[0]
    # print("===============Error----------------------")
    # if "multi_branch" not in tds_dir:
    #     print(np.round(error_stat[[0, 2, -2]], 4))
    # elif "multi_branch_z" in tds_dir:
    #     print(['%.2f' % (v*100) for v in error_stat[[0, 2, 3, -2]]])
    # elif "multi_branch_p" in tds_dir:
    #     print(['%.2f' % (v * 100) for v in error_stat[[0, 2, 3, -4, -3]]])
    if show == True:
        auc_score_tot = np.round(np.array(auc_score) * 100, 2)
        print(auc_score_tot)
        print("====================================================================")
        print("{0}         {1}".format("method", "accuracy"))
        print("{0}:    {1}".format("latent-mse", auc_score_tot[0]))
        if "fps" not in tds_dir:
            print("{0}:    {1}".format("latent-cos", auc_score_tot[1]))
            print("{0}:     {1}".format("latent-l1", auc_score_tot[2]))
            print("{0}:      {1}".format("pred-mse", auc_score_tot[3]))
            print("{0}:     {1}".format("pred-psnr", auc_score_tot[4]))       
        print("====================================================================")

    
    np.save(tds_dir + '/opt_threshold', opt_threshold)
    return auc_score


def calc_bg_ratio(stat):
    """This function calculates the background ratio,
    Args:
        stat: [num_frames, 4] score_bg_0, score_bg_1, score_bg_0+score_bg_1, score_bg_0*ratio_0+score_bg_1*ratio_1
    Return:
        bg_ratio
    """
    bg_ratio_0 = (stat[:, -1] - stat[:, 1])/(stat[:, 0] - stat[:, 1])
    bg_ratio_1 = (1 - bg_ratio_0)
    return np.mean(bg_ratio_0), np.mean(bg_ratio_1)


def give_auc_score(ano_score_tot, gt_tot):
    ano_vec = np.array([v for j in ano_score_tot for v in j])
    gt_vec = np.array([v for j in gt_tot for v in j])
    num_crit = np.shape(ano_vec)[1]
    auc_score_tot = np.zeros([num_crit])
    opt_threshold = np.zeros([num_crit])
    for single_use_index in np.arange(num_crit):
        if single_use_index == num_crit - 1 and num_crit > 3:
            gt_temp = 1.0 - gt_vec
        else:
            gt_temp = gt_vec
        _fpr, _tpr, _threshold = roc_curve(gt_temp, ano_vec[:, single_use_index])
        _auc = auc(_fpr, _tpr)
        optimal_idx = np.argmax(_tpr - _fpr)
        optimal_threshold = _threshold[optimal_idx]
        auc_score_tot[single_use_index] = _auc
        opt_threshold[single_use_index] = optimal_threshold
    return auc_score_tot, opt_threshold


def main_part_for_calc_ano_score(use_mark, tds_dir, sub_file_name, error_stat, gt_start, norm_score, gt_sub):
    score_path = tds_dir + sub_file_name
    ano_score_single_test_video = np.load(score_path)
    if np.shape(ano_score_single_test_video)[1] == 1:
        v = ano_score_single_test_video[np.where(ano_score_single_test_video[:, 0] != 0)[0], :]
        ano_score_single_test_video = v
    ano_score_renew_per_test_video = np.zeros(np.shape(ano_score_single_test_video))
    num_crit = np.shape(ano_score_single_test_video)[1]
    crit_space = np.arange(num_crit)
    for single_use_index in crit_space:
        _ano_score = ano_score_single_test_video[:, single_use_index]
        if "multi_branch" in tds_dir:
            if single_use_index in [2, 6, 10, 14]:
                _ano_score = simple_sum_ano_score(ano_score_single_test_video, single_use_index)
            elif single_use_index in [3, 7, 11, 15]:
                _ano_score = aggregate_score_by_ratio(ano_score_single_test_video, single_use_index)
        if single_use_index == num_crit - 1 and num_crit > 3:
            _ano_score = 10 * np.log10(_ano_score / ano_score_single_test_video[:, single_use_index - 1])
        if norm_score is True:
            _ano_score = (_ano_score - np.min(_ano_score)) / (np.max(_ano_score) - np.min(_ano_score))
        ano_score_renew_per_test_video[:, single_use_index] = _ano_score
    if use_mark is "pred_score":
        gt_per_video = gt_sub[-np.shape(ano_score_renew_per_test_video)[0]:]
    else:
        gt_per_video = gt_sub[gt_start:(gt_start + np.shape(ano_score_renew_per_test_video)[0])]
    return ano_score_renew_per_test_video, gt_per_video, error_stat

def aggregate_score_by_ratio(ano_score, index):
    q = ano_score[:, index]
    # bg_ratio_1 = (ano_score[:, index-3] - ano_score[:, index])/(ano_score[:, index-3]-ano_score[:, index-2])
    # bg_ratio_0 = 1-bg_ratio_1
    # norm_0 = ano_score[:, index-3]
    # norm_1 = ano_score[:, index-2]
    # # norm_0 = norm_score_func(ano_score[:, index - 3])
    # # norm_1 = norm_score_func(ano_score[:, index - 2])
    # q = norm_0*bg_ratio_0 + norm_1*bg_ratio_1
#    q = ano_score[:, index - 3]*bg_ratio_0 + ano_score[:, index - 2]*bg_ratio_1
    return q


def simple_sum_ano_score(ano_score, index):
    ano_score_update = ano_score[:, index]
    return ano_score_update


def norm_score_func(ano_score):
    ano_new = (ano_score - np.min(ano_score))/(np.max(ano_score) - np.min(ano_score))
    return ano_new

