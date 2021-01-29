import cv2
import numpy as np
from skimage.feature import daisy

def sift_des(opt):
    return cv2.xfeatures2d.SIFT_create(nfeatures=opt['nfeatures'])

def surf_des(opt):
    return cv2.xfeatures2d.SURF_create(hessianThreshold=opt['hessianThreshold'])

def create_feat_des(opt):
    des = DESCRIPTOR[opt['feat_des']]
    if des is not None:
        return des(opt)

def compute_des(img, feat_des, opt):
    if opt['feat_des'] == 'daisy':
        des = daisy(img, step=opt['step_size'])
        des = des.reshape(-1,des.shape[2])
    else:
        if opt['dense']:
            assert opt['dense_kp'] != None
            _,des = feat_des.compute(img, opt['dense_kp'])
        else:
            _,des=feat_des.detectAndCompute(img,None)

    return des

#def create_dense_kp(img_shape, step_div, num_sizes):
#    return [cv2.KeyPoint(x, y, step_size) for y in range(0, img_shape[0], step_size)
#                                          for x in range(0, img_shape[1], step_size)]

def create_dense_kp(img_shape, step_div_size=50, num_sizes=1):
    keypoints = []
    init_step_size_x = max(img_shape[1] // step_div_size, 8)
    init_step_size_y = max(img_shape[0] // step_div_size, 8)
    for i in range(1, num_sizes+1):
        current_step_size_x = init_step_size_x * i
        current_step_size_y = init_step_size_y * i
        kp_size = (current_step_size_x + current_step_size_y) // 2
        keypoints += [cv2.KeyPoint(x, y, kp_size) for y in range(0, img_shape[0], current_step_size_y)
                                                    for x in range(0, img_shape[1], current_step_size_x)]
    return keypoints

def spatial_pyramid_des_horizontal(img, feat_des, feat_des_options):
    if feat_des_options['dense']:
        feat_des_options['dense_kp'] = create_dense_kp(img.shape,
                                                       step_div_size=feat_des_options['step_div'],
                                                       num_sizes=feat_des_options['num_sizes'])

    des = compute_des(img, feat_des, feat_des_options)
    pyramid_descriptors = [des]

    # pyramid_descriptors[1:4] -> descriptors of the four cells (of size 1/4 of the image size)
    # ...

    level = feat_des_options['spatial_pyramid_level']
    for l in range(1,level+1):
        level_factor = 3*l
        cell_h = int(img.shape[0]/level_factor)

        if feat_des_options['dense']:
            feat_des_options['dense_kp'] = create_dense_kp([cell_h,img.shape[1]],
                                                            step_div_size=feat_des_options['step_div'],
                                                            num_sizes=feat_des_options['num_sizes'])
        for f_h in range(level_factor):
            shift_h = f_h*cell_h
            cell = img[shift_h:shift_h+cell_h,:]
            des = compute_des(cell, feat_des, feat_des_options)
            pyramid_descriptors.append(des)

    return pyramid_descriptors

def spatial_pyramid_des_square(img, feat_des, feat_des_options):
    if feat_des_options['dense']:
        feat_des_options['dense_kp'] = create_dense_kp(img.shape, feat_des_options['step_div'], feat_des_options['num_sizes'])

    des = compute_des(img, feat_des, feat_des_options)
    pyramid_descriptors = [des]

    level = feat_des_options['spatial_pyramid_level']
    for l in range(1,level+1):
        level_factor = 2*l
        cell_h = int(img.shape[1]/level_factor)
        cell_w = int(img.shape[0]/level_factor)

        if feat_des_options['dense']:
            feat_des_options['dense_kp'] = create_dense_kp([cell_h,cell_w], feat_des_options['step_div'], feat_des_options['num_sizes'])

        for f_h in range(level_factor):
            shift_h = f_h*cell_h
            for f_w in range(level_factor):
                shift_w = f_w*cell_w
                cell = img[shift_h:shift_h+cell_h, shift_w:shift_w+cell_w]
                des = compute_des(cell, feat_des, feat_des_options)
                pyramid_descriptors.append(des)

    return pyramid_descriptors

def spatial_pyramid_des(img, feat_des, feat_des_options):
    if feat_des_options['subregion_shape'] == 'square':
        return spatial_pyramid_des_square(img, feat_des, feat_des_options)
    elif feat_des_options['subregion_shape'] == 'horizontal':
        return spatial_pyramid_des_horizontal(img, feat_des, feat_des_options)

def spatial_pyramid_histograms(pyramid_descriptor, codebook, k):
    visual_words=np.zeros(k*len(pyramid_descriptor),dtype=np.float32)
    for d in range(len(pyramid_descriptor)):
        if pyramid_descriptor[d] is None:
            visual_words[d*k:d*k+k]=np.zeros(k)
        else:
            words=codebook.predict(pyramid_descriptor[d])
            visual_words[d*k:d*k+k]=np.bincount(words,minlength=k)
    return visual_words

DESCRIPTOR = {
    'sift': sift_des,
    'surf': surf_des,
    'daisy': None
}
