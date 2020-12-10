#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 09 17:33 2010
This script list all the hyperparameters
It includes args and also the learning rate
@author: li
"""
import argparse


def give_motion_foreground_penalty(args):    
    if args.model_type == "daml":
        print("------I am now training the SToA model DAML-------")
        motion_penalty = 0.001
        batch_size = 5
        shortcut_opt=True
    elif args.model_type == "single_branch":
        if args.data_set == "avenue":
            batch_size = 6
            fore_penalty = 0.1
            motion_penalty = 0.010
        elif args.data_set == "avenue_robust_on_rain":
            batch_size = 6
            fore_penalty = 0.4
            motion_penalty = 0.001
        elif args.data_set == "brugge":
            batch_size = 3
            fore_penalty = 0.4
            motion_penalty = 0.001
        shortcut_opt=False
    elif args.model_type == "multi_branch_z":
        if args.data_set == "avenue":
            batch_size = 4
            motion_penalty = 0.010
            fore_penalty = 0.4
        elif args.data_set == "avenue_robust_on_rain":
            batch_size = 4
            fore_penalty = 0.4
            motion_penalty = 0.001
        shortcut_opt=False
    args.batch_size = batch_size
    args.shortcut_opt=shortcut_opt
    args.fore_penalty = fore_penalty
    args.motion_penalty = motion_penalty
    return args
    
    
def get_args():
    parser = argparse.ArgumentParser(description='Multi-branch with sum shortcut')
    parser.add_argument('--datadir', type=str, help="the location that saves the data")
    parser.add_argument('--expdir', type=str, help="the location that saves the experiments")
    parser.add_argument('-ds', '--data_set', type=str, default="avenue", metavar='DATA_SET',
                        help='dataset')
    
    parser.add_argument('-bs', '--batch_size', type=int, default=4, metavar='BATCH_SIZE',
                        help='input batch size for training (default: 100)')
    parser.add_argument('-ep', '--max_epoch', type=int, default=50, metavar='EPOCHS',
                        help='maximum number of epochs')
    
    parser.add_argument('--model_type', type=str, help="which model am I using?")
    parser.add_argument('-ne', '--num_encode_layer', type=int, default=4, metavar='NUM_ENCODER_LAYER',
                        help='the number of encoder layers')
    parser.add_argument('-nd', '--num_decode_layer', type=int, default=4, metavar='NUM_DECODER_LAYER',
                        help='the number of decoder layers')
    parser.add_argument('-no', '--norm', type=bool, default=False, metavar='NORM',
                        help='whether the input is normalized')
    parser.add_argument('-sc', '--shortcut_connection', type=bool, default=True, metavar='SHORTCUT_CONNECTION',
                        help='whether I am using shortcut connection')
    parser.add_argument('-od', '--output_dim', type=int, default=3, metavar='OUTPUT_DIM',
                        help='the output dimension')
    
    parser.add_argument('-ts', '--time_step', type=int, default=6, metavar='TIME_STEP',
                        help='the number of input frames')
    parser.add_argument('-in', '--single_interval', type=int, default=2, metavar='SINGLE_INTERVAL',
                        help='the gap between every two frames')
    parser.add_argument('-de', '--delta', type=int, default=6, metavar='DELTA',
                        help='the gap between last input frame and output frame')
    parser.add_argument('-cc', '--concat_option', type=str, default="conc_tr", metavar='CONCAT_OPTION',
                        help='the concatenation method')
    
    parser.add_argument('-dv', '--darker_value', type=float, default=0.3, metavar='DARKER_VALUE',
                        help='the darkest degree of the input frame')
    parser.add_argument('-dt', '--darker_type', type=str, default="auto_all", metavar='DARKER_TYPE',
                        help='the darker type, either auto or manu')
    parser.add_argument('-ao', '--aug_opt', type=str, default="add_dark", metavar='AUG_OPT',
                        help='augmentation method, either add_dark or add_snow')
    parser.add_argument('-npbg', '--num_pred_layer_for_bg', type=int, default=4, metavar='NUM_PRED_LAYER_FOR_BG',
                        help='the number of conv layers for predicting ratio for background')
    parser.add_argument('-nbg', '--num_bg', type=int, default=2, metavar='NUM_BG',
                        help='the number of background')
    parser.add_argument('-nf', '--num_frame', type=int, default=1, metavar='NUM_FRAME',
                        help='the number of input frames')
    parser.add_argument('-neb', '--num_encoder_block', type=int, default=2, metavar='NUM_ENCODER_BLOCK',
                        help='the number of encoder blocks')
    parser.add_argument('-ps', '--pool_size', type=int, default=2, metavar='POOL_SIZE',
                        help='the stride for the encoder and decoder')
    parser.add_argument('-vs', '--version', type=int, default=1, metavar='VERSION', help="experiment version")

    
    parser.add_argument('--test_opt', type=str, help="what kind of operations do I need at test time?")
    parser.add_argument('--test_index_use', type=str, help="which sequence of data that I am going to test?")
    parser.add_argument('--rain_type', type=str, help="which rain am I using?")
    parser.add_argument('--brightness', type=int, help="which brightness", default=2)
    args = parser.parse_args()
    return args

