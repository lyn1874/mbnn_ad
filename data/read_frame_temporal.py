#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 16:13:36 2019
Prepare input for training and testing
@author: li
"""
import tensorflow as tf
import os
import numpy as np
import cv2
import scipy.io as io


class get_video_data(object):
    def __init__(self, path_for_data, data_set):
        super(get_video_data, self).__init__()
        self.model_mom = path_for_data
        self.data_set = data_set

    def read_avenue_data(self):
        path_mom = self.model_mom + "Avenue/frames/"
        for tr_or_tt in ["training", "testing"]:
            path = path_mom + tr_or_tt
            if os.path.exists(path):
                all_path = os.listdir(path)
                all_path = sorted(all_path, key=lambda s: int(s.strip().split('frame_')[1].strip().split('.jpg')[0]))
                all_path = sorted(all_path, key=lambda s: int(s.strip().split('video_')[1].strip().split('_frame')[0]))
                all_path = [path + '/' + v for v in all_path if 'jpg' in v]
            else:
                all_path = []
                print("*****PATH DOESN'T EXIST, PROBABLY WILL THROW ERROR LATER*****", path)
            if tr_or_tt is "training":
                tr_tot = all_path
            elif tr_or_tt is "testing":
                tt_tot = all_path
        print("there are %d training images and %d test images" % (np.shape(tr_tot)[0], np.shape(tt_tot)[0]))
        imshape = np.array([360, 640, 3])
        targshape = np.array([128, 224, 3])
        return np.array(tr_tot), np.array(tt_tot), imshape, targshape
    
    def read_avenue_robust_on_bright(self):
        path_mom = self.model_mom + "Avenue/frames/"
        path_tr_tot = []
        bright_space = ["original_training/bright_%.2f" % (i/10) for i in range(12)[2:]]
        for single_tr in bright_space:
#         for single_tr in ["original_training/bright_1.00", "original_training/bright_0.30", "original_training/bright_0.70"]:
            path = path_mom + single_tr
            all_path = os.listdir(path)
            all_path = sorted(all_path, key=lambda s: int(s.strip().split('frame_')[1].strip().split('.jpg')[0]))
            all_path = sorted(all_path, key=lambda s: int(s.strip().split('video_')[1].strip().split('_frame')[0]))
            all_path = [path + '/' + v for v in all_path if 'jpg' in v]
            path_tr_tot.append(all_path)
        path_tr_tot = [v for j in path_tr_tot for v in j]
        path = path_mom + "testing"
        all_path = os.listdir(path)
        all_path = sorted(all_path, key=lambda s: int(s.strip().split('frame_')[1].strip().split('.jpg')[0]))
        all_path = sorted(all_path, key=lambda s: int(s.strip().split('video_')[1].strip().split('_frame')[0]))
        path_tt = [path + '/' + v for v in all_path if 'jpg' in v]
        imshape = np.array([128, 224, 3])
        targshape = np.array([128, 224, 3])
        return np.array(path_tr_tot), np.array(path_tt), imshape, targshape
        
    def read_avenue_robust_on_rain(self):
        path_mom = self.model_mom + "Avenue/frames/"
        path_tr_tot = []
        for single_tr in ["training", "heavy_training/bright_0.70", "torrential_training/bright_0.70"]:
            path = path_mom + single_tr
            all_path = os.listdir(path)
            all_path = sorted(all_path, key=lambda s: int(s.strip().split('frame_')[1].strip().split('.jpg')[0]))
            all_path = sorted(all_path, key=lambda s: int(s.strip().split('video_')[1].strip().split('_frame')[0]))
            all_path = [path + '/' + v for v in all_path if 'jpg' in v]
            path_tr_tot.append(all_path)
        path_tr_tot = [v for j in path_tr_tot for v in j]

        path = path_mom + "testing"
        all_path = os.listdir(path)
        all_path = sorted(all_path, key=lambda s: int(s.strip().split('frame_')[1].strip().split('.jpg')[0]))
        all_path = sorted(all_path, key=lambda s: int(s.strip().split('video_')[1].strip().split('_frame')[0]))
        path_tt = [path + '/' + v for v in all_path if 'jpg' in v]
        imshape = np.array([360, 640, 3])
        targshape = np.array([128, 224, 3])
        return np.array(path_tr_tot), np.array(path_tt), imshape, targshape
    
    def read_avenue_augmented_test_data(self, rain_type, brightness):
        path_mom = self.model_mom + "Avenue/frames/"
        if brightness > 1:
            brightness = brightness / 10
        path_tt_mom = path_mom + "%s_testing/bright_%.02f/" % (rain_type, brightness)
        all_path = [v for v in os.listdir(path_tt_mom) if '.jpg' in v]
        all_path = sorted(all_path, key=lambda s: int(s.strip().split('frame_')[1].strip().split('.jpg')[0]))
        all_path = sorted(all_path, key=lambda s: int(s.strip().split('video_')[1].strip().split('_frame')[0]))
        imshape = np.array([360, 640, 3])
        targshape = np.array([128, 224, 3])
        return [], np.array([path_tt_mom + v for v in all_path]), imshape, targshape        

    def read_brugge_data(self):
        path_tr_mom = self.model_mom + 'brugge/'
        path_tr_tot = []
        for date in ["Nov_27", "Nov_29"]:
            _path = path_tr_mom + date
            seq = sorted(os.listdir(_path))
            for single_seq in seq:
                _seqpath = _path + "/" + single_seq + "/"
                path_tr_tot.append([_seqpath + v for v in sorted(os.listdir(_seqpath)) if '.jpg' in v])
        path_tr_tot = [v for j in path_tr_tot for v in j]
        path_tt_tot = path_tr_tot[-100:]

        imshape = np.array([1024, 1280, 3])
        targshape = np.array([320, 400, 3])
        return np.array(path_tr_tot), np.array(path_tt_tot), imshape, targshape

    def forward(self, tr_time="Nov_21", rain_type=None, brightness=0):
        print(self.data_set, tr_time)
        if "avenue" in self.data_set and tr_time == None:
            if rain_type != None and rain_type != "None":
                return self.read_avenue_augmented_test_data(rain_type, brightness)
            else:
                print("Returning the original dataset...........")
                return self.read_avenue_data()
        elif "avenue" in self.data_set and tr_time == "robust_on_rain":
            return self.read_avenue_robust_on_rain()
        elif "avenue" in self.data_set and tr_time == "robust_on_bright":
            return self.read_avenue_robust_on_bright()
        elif self.data_set == "brugge":
            return self.read_brugge_data()
        else:
            print("===============the required dataset doesn't exist yet===============")


def read_data(args, path_for_data, tr_time="Nov_21", test_index_use=None):
    if args.data_set != "shanghaitech":
        train_im, test_im, imshape, targ_shape = get_video_data(path_for_data, 
                                                                args.data_set).forward(tr_time,
                                                                                       args.rain_type, args.brightness)
        if test_index_use:
            im_use = np.array([v for v in test_im if test_index_use in v])
        else:
            im_use = train_im
        train_im_interval, in_shape, out_shape = read_frame_interval_by_dataset(args.data_set, im_use, args.time_step,
                                                                                args.concat_option,
                                                                                [args.single_interval], args.delta)
    return im_use, train_im_interval, imshape, targ_shape, in_shape, out_shape


def read_frame_interval_by_dataset(data_set, tr_im, time_step, concat, interval, delta=None, neg=False):
    if concat == "conc_tr":
        print("=================================")
        print("There are %d input with %d interval between each of them" % (time_step, interval[0]))
        print("These input are used to predict frame at t+%d" % delta)
        print("================================")
    if data_set == "avenue":
        video_index = [v.strip().split('/frames/')[1].strip().split('frame_')[0] for v in tr_im]
        name_space = np.unique(video_index)
        name_space = [v + 'frame_' for v in name_space]
        num_test_video = np.shape(name_space)[0]
        #        name_space = ['_video_' + v + '_frame_' for v in name_space]
        print("There are %d videos for dataset %s" % (num_test_video, name_space))
        tr_temporal_tot = []
        for single_index in range(num_test_video):
            name_use = name_space[single_index]
            tr_subset = sorted([v for v in tr_im if name_use in v],
                               key=lambda s: int(s.strip().split(name_use)[-1].strip().split('.jpg')[0]))
            tr_subset = np.array(tr_subset)
            tr_temp, in_shape, out_shape = read_frame_interval(tr_subset, time_step, concat, interval, delta,
                                                               neg=neg)
            tr_temporal_tot.append(tr_temp)
        tr_temporal_tot = np.array([y for x in tr_temporal_tot for y in x])
        print("the shape of concatenate tr path", np.shape(tr_temporal_tot))
    elif data_set == "brugge":
        unique_name = ['/'.join(v.strip().split('brugge')[1].strip().split('/')[:3]) for v in tr_im]
        name_space = np.unique(unique_name)
        num_test_video = np.shape(name_space)[0]
        print("There are videos", num_test_video)
        print("The unique name", name_space)
        tr_temporal_tot = []
        for single_video_name in name_space:
            tr_subset = sorted([v for v in tr_im if single_video_name in v])
            tr_subset = np.array(tr_subset)
            tr_temp, in_shape, out_shape = read_frame_interval(tr_subset, time_step, concat, interval, delta,
                                                               neg=neg)
            tr_temporal_tot.append(tr_temp)
        tr_temporal_tot = np.array([y for x in tr_temporal_tot for y in x])
        print("the shape of concatenate tr path", np.shape(tr_temporal_tot))
    else:
        tr_subset = tr_im
        tr_temp, in_shape, out_shape = read_frame_interval(tr_subset, time_step, concat, interval, delta, neg=neg)
        tr_temporal_tot = np.array(tr_temp)
    return tr_temporal_tot, in_shape, out_shape


def read_tensor_and_aug_in_npy_value(im_original, aug_opt, rain_type, dark_value,
                                     time_step, interval, delta):
    """We directly apply the augmentation on the npy array instead of using tensorflow, because it's
    way to slow to use the tensorflow version
    Ops:
    1. I need to load the data in a npy version, then I need to change them in the RGB
    2. Then I apply the augmentation as required, such as adding rain or adding dark with the specific
    dark value
    3. Then I need to divide them by 255.0
    4. Then I need to resize them to the target shape
    5. Then I will output the augmented dataset
    """
#    im_original = [cv2.imread(v)[:, :, ::-1] for v in im_filename]
    im_new = []
    if aug_opt == "add_dark":
        for iterr, single_im in enumerate(im_original):
            single_im = darker_npy(single_im, dark_value)/255.0
            im_new.append(single_im)
    elif aug_opt == "add_rain" and rain_type == "torrential":
        for iterr, single_im in enumerate(im_original):
            single_im = add_rain_torrential(single_im, dark_value)/255.0
            im_new.append(single_im)
    elif aug_opt == "add_rain" and rain_type == "heavy":
        for iterr, single_im in enumerate(im_original):
            single_im = add_rain_heavy(single_im, dark_value)/255.0
            im_new.append(single_im)
    im_new = np.array(im_new)
    print(np.shape(im_new))
    im_new_frame_interval, inshape, outshape = read_frame_interval(im_new, time_step, "conc_tr", interval, delta)
    return np.array(im_new_frame_interval)


def read_frame_interval(all_cat, time_step, concat, interval, delta, bg=None, neg=False):
    all_cat_new = []
    input_is_filename = isinstance(all_cat[0], str)
    if input_is_filename is True:
        append_value = '0'
    else:
        append_value = np.zeros(shape=np.shape(all_cat[0]))
    for single_interval in interval:
        all_cat_use = all_cat
        num_cat = np.shape(all_cat_use)[0]
        crit = single_interval * (time_step - 1) + delta
        for i, v in enumerate(all_cat_use):
            if i + crit < num_cat:
                init = np.linspace(i, i + single_interval * (time_step - 1), time_step, dtype='int32')
                end = init[-1] + delta
                if bg:
                    tt = np.concatenate([[all_cat_use[end]],
                                         [bg],
                                         [append_value for i in range(time_step - 2)]], axis=0)
                else:
                    tt = np.concatenate([[all_cat_use[end]],
                                         [append_value for i in range(time_step - 1)]], axis=0)
                # tt = np.repeat(all_cat_use[end], repeats = time_step)
                if input_is_filename is True:
                    all_cat_new.append([all_cat_use[init], tt])
                else:
                    all_cat_new.append(np.concatenate([all_cat_use[init], [all_cat_use[end]]],
                                                      axis=0))
    inp_shape = np.shape(all_cat_new[0][0])
    oup_shape = np.shape(all_cat_new[1][0])

    return all_cat_new, inp_shape, oup_shape


def rgb2hls(image):
    image = np.array(image)
    im = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    return im


def hls2rgb(image):
    image = np.array(image)
    im = cv2.cvtColor(image, cv2.COLOR_HLS2RGB)
    return im


def darker_tf(image, darker_value):
    """this function is a tensorflow version of darker image.
    The image can be [num_frame,batch_size, imh, imw, ch], or just [batch_size,imh,imw,ch]
    or only [imh, imw, ch]
    It needs to be float32, in the range of [0,1] [rgb]
    1) Transfer the image to hsv       
    2) unscatter based on the last channel v, and multiple v with the darker value 
    3) Then clip the last to make sure the maximum value is 1.0, and min is 0.0
    4) concate them based on the last channel
    5) transfer it back to rgb
    """
    hls_image = tf.py_function(rgb2hls, inp=[image], Tout=tf.uint8)
    hls_image = tf.cast(hls_image, tf.float32)
    hls_image = tf.unstack(hls_image, num=3, axis=-1, name="unstack_hls")
    v = tf.multiply(hls_image[1], darker_value)
    v = tf.clip_by_value(v, clip_value_min=0.0, clip_value_max=255.0)
    hls_image[1] = v
    hls_new = tf.stack(hls_image, axis=-1, name="stack_hls")
    hls_new = tf.cast(hls_new, tf.uint8)
    rgb_new = tf.py_function(hls2rgb, inp=[hls_new], Tout=tf.uint8)
    return rgb_new


def darker_npy(image, darker_value):
    hls_image = rgb2hls(image)
    hls_image = np.array(hls_image, dtype=np.float32)
    hls_image[:, :, 1] = hls_image[:, :, 1] * darker_value
    hls_image[:, :, 1] = hls_image[:, :, 1].clip(0, 255.0)
    hls_image = np.array(hls_image, dtype=np.uint8)
    rgb_image = hls2rgb(hls_image)
    return np.array(rgb_image, dtype=np.float32)


def small_func(v, imshape, targ_shape, darker_value, augment_option, crop_im):
    v = tf.image.decode_jpeg(tf.read_file(v), channels=3)
    v = tf.reshape(v, shape=imshape)
    if augment_option == "add_dark":
        print("During training, randomly adjust the brightness of the images")
        v = darker_tf(v, darker_value)
    elif augment_option == "none":
        v = v
    if imshape[0] != targ_shape[0]:
        v = tf.expand_dims(tf.cast(v, tf.float32), axis=0)
        if crop_im == True:
            v = tf.image.crop_to_bounding_box(v, 150, 0, 930, 1500)
        v = tf.image.resize_bilinear(tf.divide(v, 255.0), targ_shape[:2])
        v = tf.squeeze(v, axis=0)
    else:
        v = tf.divide(tf.cast(v, tf.float32), 255.0)
    v = tf.reshape(v, shape=targ_shape)
    return v


def concat_im_directly_on_filenames(path_for_load_data, data_set, x, imshape, targshape, learn_opt,
                                    train_index, darker_value, aug_opt, darker_type, crop_im):
    """This function adds the augmentation directly on each of the frames instead of on each of sequential
    frames
    x: [filename length]
    """
    x = tf.reshape(x, [1])
    inp_im = tf.map_fn(lambda v: small_func(v, imshape, targshape, darker_value, 
                                            aug_opt, crop_im), x,
                       dtype=tf.float32)
    if "learn_rest" in learn_opt or "learn_fore" in learn_opt:
        print("loading mean from npy")
        data_mean = calc_mean_std_data(path_for_load_data, data_set, train_index,
                                       targ_shape=targshape)
        inp_im = tf.subtract(inp_im, data_mean)
    else:
        data_mean = tf.constant(0.0)
    im_tot = [inp_im, data_mean]
    return im_tot


def concat_im(path_for_load_data, data_set, x, inp_shape, oup_shape, imshape, targ_shape, learn_opt,
              train_index, darker_value, aug_opt, darker_type, crop_im):
    """this function is used to concatenate the input and output
    Args:
        path_for_load_data: the mom path that saves the data
        data_set: str, "avenue" or "antwerpen"
        x: string tensor, the shape is [2]
        inp_shape: [time_step] define the number of input frames 
        oup_shape: [num_output] define the number of output frames
        imshape: [imh, imw, ch] np.array
        targ_shape: [imh, imw, ch], np.array
        learn_opt: learn_full or learn_fore (subtract background)
        train_index: only for shanghaitech dataset
        darker_value: a npy value, can be either a list or a single number
        aug_opt: "add_dark" or "none"
        darker_type: if it's manu: then all the training frames use the same darker degree
        if it's "auto", then some of them use 1.0 which don't apply darker adjustness, the rest of them use 
        darker_value degree
        if it's auto_all, then the model has seen all types of brightness during training
        crop_im: bool. True/False. When I train the antwerpen dataset, it needs to be True, Otherwise, it's always
                 False
    """
    inp = tf.reshape(x[0], inp_shape)
    oup_tot = tf.reshape(x[1], oup_shape)
    oup = oup_tot[:1]
    if aug_opt != "add_rain":
        if darker_type == "auto":
            darker_space = [1.3, 1.0, darker_value]
            darker_value = tf.random.shuffle(darker_space)[0]
        elif darker_type == "auto_all":
            darker_space = list(np.linspace(0.3, 1.1, 9))
            darker_value = tf.random.shuffle(darker_space)[0]
        else:
            print("I am manually select coef")
            darker_value = darker_value
    if "auto" in darker_type:
        print("Performing augmentation during training", aug_opt, darker_space, darker_value)
        
    inp_im = tf.map_fn(lambda v: small_func(v, imshape, targ_shape, darker_value, 
                                            aug_opt, crop_im), inp,
                       dtype=tf.float32)
    out_im = tf.map_fn(lambda v: small_func(v, imshape, targ_shape, darker_value, 
                                            aug_opt, crop_im), oup,
                       dtype=tf.float32)  # num_conc_image, imh, imw, ch
    if train_index != None:
        if "learn_rest" in learn_opt or "learn_fore" in learn_opt:
            print("loading mean from npy")
            data_mean = calc_mean_std_data(path_for_load_data, data_set, train_index,
                                           targ_shape=targ_shape)
            inp_im = tf.subtract(inp_im, data_mean)
            out_im = tf.subtract(out_im, data_mean)
        else:
            data_mean = tf.constant(0.0)
    else:
        data_mean = tf.constant(0.0)

    im_tot = [inp_im, out_im, data_mean]
    return im_tot


def calc_mean_std_data(path_for_load_data, data_set, tr_index=None, tensor=True, targ_shape=np.array([128, 196, 3])):
    path4bg = "gt/"
    if tr_index == None:
        path2read = path4bg + "%s_mean_%d_%d_auto_%.1f.npy" % (data_set, targ_shape[0], targ_shape[1], 0.2)
    elif "robust" in tr_index:
        path2read = path4bg + "%s_mean_%d_%d_%s.npy" % (data_set, targ_shape[0], targ_shape[1], tr_index)
    print("-----Loading the pre-calculated background from path------------------")
    print(path2read)
    mean_value = np.load(path2read)
    mean_value = np.expand_dims(mean_value, axis=0)  # [batch_size, imh, imw, ch] ch=1/3
    if tensor is True:
        return tf.constant(mean_value, dtype=tf.float32)
    else:
        return mean_value
    

def dataset_input(path_for_load_data, data_set, im_filename, learn_opt, temp_shape, imshape, targ_shape, batch_size,
                  shuffle, train_index=None, epoch_size=1, darker_value=0, aug_opt="none",
                  darker_type="manu", crop_im=False):
    """This function is used to read the images 
    """
    print("=================================================")
    print("The applied aug method", aug_opt)
    print("The darker value", darker_value, "darker_type", darker_type)
    print("=================================================")

    inp_shape, oup_shape = temp_shape
    images = tf.convert_to_tensor(im_filename, dtype=tf.string)
    if shuffle == True:
        images = tf.random.shuffle(images)

    transform1 = tf.data.Dataset.from_tensor_slices(images)
    transform1 = transform1.repeat(epoch_size)
    transform2 = transform1.map(lambda x: concat_im(path_for_load_data,
                                                    data_set,
                                                    x, inp_shape, oup_shape,
                                                    imshape=imshape,
                                                    targ_shape=targ_shape,
                                                    learn_opt=learn_opt,
                                                    train_index=train_index,
                                                    darker_value=darker_value,
                                                    aug_opt=aug_opt,
                                                    darker_type=darker_type,
                                                    crop_im=crop_im))
    transform2 = transform2.apply(tf.data.experimental.ignore_errors())
    im_out = transform2.batch(batch_size, drop_remainder=True)
    return im_out


def dataset_test_input(path_for_load_data, data_set, im_filename, learn_opt, imshape, targ_shape,
                       batch_size, shuffle, train_index=None, epoch_size=1, darker_value=0, 
                       aug_opt="none", darker_type="manu", crop_im=False):
    """This function is used to read the images
    """
    print("=================================================")
    print("The applied aug method", aug_opt)
    print("The darker value", darker_value, "darker_type", darker_type)
    print("=================================================")
    images = tf.convert_to_tensor(im_filename, dtype=tf.string)
    if shuffle == True:
        images = tf.random.shuffle(images)
    transform1 = tf.data.Dataset.from_tensor_slices(images)
    transform1 = transform1.repeat(epoch_size)
    transform2 = transform1.map(lambda x: concat_im_directly_on_filenames(path_for_load_data, data_set, x,
                                                                          imshape, targ_shape, learn_opt, train_index,
                                                                          darker_value, aug_opt,
                                                                          darker_type, crop_im))
    transform2 = transform2.apply(tf.data.experimental.ignore_errors())
    im_out = transform2.batch(batch_size, drop_remainder=False)
    return im_out


def read_tensor(args, path_for_load_data, data_set, targ_shape, imshape, temp_shape, train_or_test, tr_time=None,
                batch_size=None):
    """This function is for loading input tensor
    rain_type: tf.placheolder"""
    if train_or_test == "train" or train_or_test == "test":
        placeholder_shape = [None, 2, temp_shape[0][0]]
    else:
        placeholder_shape = [None, 1]
    if train_or_test == "train":
        shuffle_option = True
        if "project" in path_for_load_data:
            if "brugge" in data_set:
                repeat = 1 #20
            else:
                repeat = 20
        else:
            repeat = 1
        darker_value = args.darker_value
        print("--------------I am loading training data with %s and repeat %d" % (shuffle_option, repeat))
    elif train_or_test == "test" or train_or_test == "test_for_score":
        shuffle_option = False
        repeat = 1
        darker_value = tf.placeholder(tf.float32, name="dark_degree")
        print("--------------I am loading test data with %s and repeat %d" % (shuffle_option, repeat))
    images_in = tf.placeholder(tf.string, shape=placeholder_shape, name='tr_im_path')
    targ_shape_for_loading = [targ_shape[0], targ_shape[1], 3]
    if train_or_test == "train" or train_or_test == "test":
        image_queue = dataset_input(path_for_load_data, data_set, images_in, args.learn_opt,
                                    temp_shape, imshape, targ_shape_for_loading, args.batch_size,
                                    shuffle=shuffle_option,
                                    train_index=tr_time,
                                    epoch_size=repeat,
                                    darker_value=darker_value,
                                    aug_opt=args.aug_opt,
                                    darker_type=args.darker_type,
                                    crop_im=args.crop_im)
    else:
        image_queue = dataset_test_input(path_for_load_data, data_set, images_in, args.learn_opt,
                                         imshape, targ_shape_for_loading, batch_size,
                                         shuffle_option, tr_time, repeat, darker_value,
                                         args.aug_opt, args.darker_type,  args.crop_im)
    image_init = image_queue.make_initializable_iterator()
    image_batch = image_init.get_next()
    if train_or_test == "train" or train_or_test == "test":
        x_input = image_batch[0]  # [batch_size, num_input_channel, imh, imw, ch]
        x_output = image_batch[1]  # [batch_size, self.output_dim, imh, imw, ch]
        x_background = image_batch[-1]
        x_input = tf.concat([x_input, x_output], axis=1)  # [batch_size, num_input+num_predict, imh, imw, ch]
        print("=========================================")
        print("The input of the model", x_input)
        print("The output of the model", x_output)
        print("=========================================")
        x_input = tf.transpose(x_input, perm=(1, 0, 2, 3, 4))  # num_frame, batch_size, imh, imw, ch
        if "learn_full" in args.learn_opt:
            x_real_input = x_input
        else:
            x_background = tf.transpose(x_background, perm=(1, 0, 2, 3, 4))  # [num_frame,batch_size, imh, imw, ch]
            x_real_input = x_input + x_background
    else:
        x_input = image_batch[0]
        x_background = image_batch[-1]
        if "learn_full" in args.learn_opt:
            x_real_input = x_input
        else:
            x_real_input = x_input + x_background
    return images_in, x_input, image_init, x_background, x_real_input, darker_value
