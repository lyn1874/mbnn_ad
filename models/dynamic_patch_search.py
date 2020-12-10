#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 18:03:32 2018
This file is the new version for calculating the uncertainty value in each patch
It's better because:
    1. It's a dynamic way of chosing most uncertain patch, since the provided patch can have overlapping for the adjacent pixels
    2. It can be further developed to have the weighted uncertain for each patch by 1/(h*w) where h and w are the height and
    width of the patch.
The thing I needs to be careful about this is that:
    1. The selected most uncertain patch needs to be able to transformed back to the binary mask
    2. The uncertainty value for the previously selected patch needs to be not considered during the selection. I think I can still set a 
    fixed number of patches, it's just it will be much more than before.

@author: s161488
"""
import numpy as np
from scipy import signal


def calculate_score_for_patch(image, kernel, stride_size, Num_Most_Uncert_Patch, crit = None, higher = True):
    """This function is used to calculate the utility score for each patch.
    Args:
        uncertainty_est: [Im_h, Im_w]
        kernel: [k_h, k_w]
    Returns:
        most_uncert_image_index: [Num_Most_Selec] this should be the real image index
        %most_uncert_patch_index: [Num_Most_Selec] this should be the numeric index for the selected patches
        binary_mask: [Num_Most_Selec, Im_h, Im_w,1]
        %pseudo_label: [Num_Most_Selec, Im_h, Im_w,1]
    Op:
        Before, I enter the uncert_est, I need to consider if there are already selected patches in the last acquisition step.
        If there are some selected patches in the last acquisition step, then it can be annotated by the binary mask. Therefore, 
        before I enter the uncert_est, the uncertainty value for the selected patches should be zero. 
        Then the evaluation for the rest patches will be as same as below
        Also, another thing needs to be considered is that if there are overlapping betweeen the new selected images and the previously
        selected images, I need to aggregate the binary mask, as same as the ground truth label. This step will be as same as before.
    """
    Im_h, Im_w = np.shape(image)
    kh, kw = np.shape(kernel)
    h_num_patch = Im_h-kh+1
    w_num_patch = Im_w-kw+1
    num_row_wise = h_num_patch//stride_size
    num_col_wise = w_num_patch//stride_size
    if stride_size == 1:
        tot_num_patch_per_im = num_row_wise*num_col_wise
    else:
        tot_num_patch_per_im = (num_row_wise+1)*(num_col_wise+1)
    patch_tot = select_patches_in_image_area(image, kernel, stride_size, num_row_wise, num_col_wise)
    patch_tot = np.reshape(patch_tot, [-1])    
    #print('Based on the experiments, there are %d patches in total'%np.shape(patch_tot)[0])
    #print('Based on the calculation, there supposed to be %d patches in tot'%(Num_Im*tot_num_patch_per_im))
    sorted_index = np.argsort(patch_tot)
    if higher is True:
        select_most_uncert_patch = (sorted_index[-Num_Most_Uncert_Patch:]).astype('int64')
    else:
        select_most_uncert_patch = (sorted_index[:Num_Most_Uncert_Patch]).astype('int64')
    if crit is not None:
        select_most_uncert_patch = (sorted_index[np.array(sorted(patch_tot))>=crit]).astype('int64')
        Num_Most_Uncert_Patch = np.shape(select_most_uncert_patch)[0]
    if Num_Most_Uncert_Patch > 0:
#        Num_Most_Uncert_Patch = np.shape(select_most_uncert_patch)[0]
        select_most_uncert_patch_index_per_im = (select_most_uncert_patch%tot_num_patch_per_im).astype('int64')
        if stride_size == 1:
            select_most_uncert_patch_rownum_per_im = (select_most_uncert_patch_index_per_im//num_col_wise).astype('int64')
            select_most_uncert_patch_colnum_per_im = (select_most_uncert_patch_index_per_im%num_col_wise).astype('int64')
        else:
            select_most_uncert_patch_rownum_per_im = (select_most_uncert_patch_index_per_im//(num_col_wise+1)).astype('int64')
            select_most_uncert_patch_colnum_per_im = (select_most_uncert_patch_index_per_im%(num_col_wise+1)).astype('int64')
        transfered_rownum, transfered_colnum = transfer_strid_rowcol_backto_nostride_rowcol(select_most_uncert_patch_rownum_per_im,
                                                                                            select_most_uncert_patch_colnum_per_im,
                                                                                            [h_num_patch, w_num_patch],
                                                                                            [num_row_wise+1, num_col_wise+1],
                                                                                            stride_size)
        binary_mask_tot = []
        box_coord = np.zeros([Num_Most_Uncert_Patch, 4])
        for i in range(Num_Most_Uncert_Patch):
            single_binary_mask = generate_binary_mask(Im_h, Im_w, 
                                                      transfered_rownum[i],
                                                      transfered_colnum[i],
                                                      kh, kw)
            row, col = np.where(single_binary_mask!=0)
            row_sort = sorted(row)
            col_sort = sorted(col)
            box_coord[i,:] = [row_sort[0], col_sort[0], row_sort[-1], col_sort[-1]]
            binary_mask_tot.append(single_binary_mask)
            #    binary_mask_tot = np.sum(binary_mask_tot, axis = 0)
        binary_mask_tot = (np.sum(binary_mask_tot, axis = 0)!=0).astype('int32')
        box_coord = np.array(box_coord, dtype = np.int32)
    else:
        binary_mask_tot = np.zeros([Im_h, Im_w], dtype = np.int32)
        box_coord = np.zeros([1, 4], dtype = np.int32)
    return binary_mask_tot, box_coord

def test_calc_patch():
    import matplotlib.pyplot as plt
    im = np.random.random([128,192])
    kernel = np.ones([10,10])
    stride = 5
    binary_mask_tot, box_coord = calculate_score_for_patch(im, kernel, stride, 7)
    
    for iterr, single_binary in enumerate(binary_mask_tot):
        fake_bi = np.zeros([128,192])
        single_coord = box_coord[iterr]
        fake_bi[single_coord[0]:(single_coord[2]+1), single_coord[1]] = 1
        fake_bi[single_coord[0]:(single_coord[2]+1), single_coord[-1]] = 1
        fake_bi[single_coord[0], single_coord[1]:(single_coord[-1]+1)] = 1
        fake_bi[single_coord[2], single_coord[1]:(single_coord[-1]+1)] = 1
        diff = np.sum(fake_bi-single_binary)
        print(iterr, diff)
        fig = plt.figure(figsize = (6,3))
        ax = fig.add_subplot(121)
        ax.imshow(single_binary)
        ax = fig.add_subplot(122)
        ax.imshow(fake_bi)
    

def return_pseudo_label(single_gt, single_fb_pred, single_binary_mask):
    """This function is used to return the pseudo label for the selected patches in per image
    Args:
        single_gt: [Im_h, Im_w,1]
        fb_pred:[Im_h, Im_w,2]
        ed_pred:[Im_h, Im_w,2]
        binary_mask: [Im_h, Im_w]
    Return:
        pseudo_fb_la: [Im_h, Im_w, 1]
        pseudo_ed_la: [Im_h, Im_w, 1]
    """
    single_gt = single_gt.astype('int64')
    fake_pred = np.argmax(single_fb_pred, axis = -1).astype('int64')
    pseudo_fb_la = fake_pred*(1-single_binary_mask)+single_gt*single_binary_mask
    return pseudo_fb_la

def generate_binary_mask(Im_h, Im_w, rowindex, colindex, kh, kw):
    """This function is used to generate the binary mask for the selected most uncertain images
    Args:
        Im_h, Im_w are the size of the binary mask
        row_index, col_index are the corresponding row and column index for most uncertain patch
        kh,kw are the kernel size
    Output:
        Binary_Mask
    Opts: 
        To transform from the selected patch index to the original image. It will be like
        rowindex:rowindex+kh
        colindex:colindex+kw
    """
    binary_mask = np.zeros([Im_h, Im_w])
    binary_mask[rowindex, colindex:(colindex+kw)] = 1
    binary_mask[rowindex+kh-1, colindex:(colindex+kw)] = 1
    binary_mask[rowindex:(rowindex+kh), colindex] = 1
    binary_mask[rowindex:(rowindex+kh), colindex+kw-1] = 1
#    binary_mask[rowindex:(rowindex+kh), colindex:(colindex+kw)] = 1
    return binary_mask

def transfer_strid_rowcol_backto_nostride_rowcol(rownum,colnum,no_stride_row_col, stride_row_col, stride_size):
    """This function is used to map the row index and col index from the strided version back to the original version
    if the row_num and col_num are not equal to the last row num or last col num
    then the transfer is just rownum*stride_size, colnum*stride_size
    but if the row_num and colnum are actually the last row num or last col num
    then the transfer is that rownum*stride_size, colnum_no_stride, or row_num_no_stride, colnum*stride_size
    """
    if stride_size != 1:
        row_num_no_stride, col_num_no_stride = no_stride_row_col
        row_num_stride, col_num_stride = stride_row_col
        transfered_row_num = np.zeros([np.shape(rownum)[0]])
        for i in range(np.shape(rownum)[0]):
            if rownum[i] != (row_num_stride-1):
                transfered_row_num[i] = stride_size*rownum[i]
            else:
                transfered_row_num[i] = row_num_no_stride-1
        transfered_col_num = np.zeros([np.shape(colnum)[0]])
        for i in range(np.shape(colnum)[0]):
            if colnum[i] != (col_num_stride-1):
                transfered_col_num[i] = colnum[i]*stride_size
            else:
                transfered_col_num[i] = col_num_no_stride-1
    else:
        transfered_row_num = rownum
        transfered_col_num = colnum
    return transfered_row_num.astype('int64'), transfered_col_num.astype('int64')
    
    
def select_patches_in_image_area(single_fb, kernel, stride_size, num_row_wise, num_col_wise):
    """There needs to be a stride"""
    utility_patches = signal.convolve(single_fb, kernel, mode = 'valid')
    if stride_size != 1:
        subset_patch = np.zeros([num_row_wise+1, num_col_wise+1])
        for i in range(num_row_wise):
            for j in range(num_col_wise):
                subset_patch[i,j] = utility_patches[i*stride_size, j*stride_size]
        for i in range(num_row_wise):
            subset_patch[i,-1] = utility_patches[i*stride_size, -1]
        for j in range(num_col_wise):
            subset_patch[-1,j] = utility_patches[-1, j*stride_size]
        subset_patch[-1,-1] = utility_patches[-1,-1]
    else:
        subset_patch = utility_patches
    return subset_patch

