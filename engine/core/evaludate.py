#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import os
import cv2
from datasets.process.heatmaps_process import get_max_preds
from datasets.zoo.posetrack import PoseTrack_COCO_Keypoint_Ordering
from utils.utils_color import COLOR_DICT


def calc_dists(preds, target, normalize):
    preds = preds.astype(np.float32)
    target = target.astype(np.float32)
    dists = np.zeros((preds.shape[1], preds.shape[0]))
    for n in range(preds.shape[0]):  # batch num
        for c in range(preds.shape[1]):  # keypoint type
            if target[n, c, 0] > 1 and target[n, c, 1] > 1:
                normed_preds = preds[n, c, :] / normalize[n]
                normed_targets = target[n, c, :] / normalize[n]
                dists[c, n] = np.linalg.norm(normed_preds - normed_targets)  # Euclidean distance
            else:
                dists[c, n] = -1
    return dists

def save_result_images(dir_out, img, pose, vis, name='', heatmaps=None, label=None):

    PoseTrack_Keypoint_Pairs = [
        # [2, 0, 'Rosy'],         # ['head_top', 'nose', 'Rosy'],
        # [0, 1, 'Rosy'],         # ['nose', 'head_bottom', 'Rosy'],
        [2, 1, 'Rosy'],  # ['head_top', 'head_bottom', 'Rosy'],
        # [0, 3, 'Rosy'],
        # [0, 4, 'Rosy'],
        [1, 6, 'Yellow'],  # ['head_bottom', 'right_shoulder', 'Yellow'],
        [1, 5, 'Yellow'],  # ['head_bottom', 'left_shoulder', 'Yellow'],
        [6, 8, 'Blue'],  # ['right_shoulder', 'right_elbow', 'Blue'],
        [8, 10, 'Blue'],  # ['right_elbow', 'right_wrist', 'Blue'],
        [5, 7, 'Green'],  # ['left_shoulder', 'left_elbow', 'Green'],
        [7, 9, 'Green'],  # ['left_elbow', 'left_wrist', 'Green'],
        [6, 12, 'Purple'],  # ['right_shoulder', 'right_hip', 'Purple'],
        [5, 11, 'SkyBlue'],  # ['left_shoulder', 'left_hip', 'SkyBlue'],
        # [11, 12, 'Yellow'],
        [12, 14, 'Purple'],  # ['right_hip', 'right_knee', 'Purple'],
        [14, 16, 'Purple'],  # ['right_knee', 'right_ankle', 'Purple'],
        [11, 13, 'SkyBlue'],  # ['left_hip', 'left_knee', 'SkyBlue'],
        [13, 15, 'SkyBlue'],  # ['left_knee', 'left_ankle', 'SkyBlue'],
    ]

    if not os.path.isdir(dir_out):
        os.makedirs(dir_out)

    img = (img - img.min()) / (img.max() - img.min()) * 255
    img = img[:,:,::-1]
    cv2.imwrite(os.path.join(dir_out, '{}img.png'.format(name)), img)
    img_pose = img.copy()
    pose = pose * [img.shape[0] / heatmaps.shape[1], img.shape[1] / heatmaps.shape[2]]
    for i in range(len(PoseTrack_Keypoint_Pairs)):
        c, p = PoseTrack_Keypoint_Pairs[i][0], PoseTrack_Keypoint_Pairs[i][1]
        if vis[c] < 0.1 or vis[p] < 0.1:
            continue
        child = tuple(pose[c].astype(int))
        parent = tuple(pose[p].astype(int))
        color = COLOR_DICT[PoseTrack_Keypoint_Pairs[i][2]]
        cv2.line(img_pose, child, parent, color, 4)

    if label is not None:
        for i in range(len(pose)):
            coord = pose[i]
            cv2.circle(img_pose, coord.astype(int), 3, (0, 255 * label[i] if label is not None else 0, 255), -1)
    cv2.imwrite(os.path.join(dir_out, '{}img_pose.png'.format(name)), img_pose)

    if heatmaps is not None:
        heatmap = np.sum(heatmaps, axis=0)
        heatmap = np.clip(heatmap * 255, 0, 255).astype(np.uint8)
        colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_BONE)
        colored_heatmap = cv2.resize(colored_heatmap, (img.shape[1], img.shape[0]))
        cv2.imwrite(os.path.join(dir_out, '{}heatmap.png'.format(name)), colored_heatmap)
        img_heatmap = img_pose * 0.7 + colored_heatmap * 0.3
        cv2.imwrite(os.path.join(dir_out, '{}img_heatmap.png'.format(name)), img_heatmap)

def save_fusion_images(dir_out, img, name='', heatmaps=None):
    if not os.path.isdir(dir_out):
        os.makedirs(dir_out)

    img = (img - img.min()) / (img.max() - img.min()) * 255
    for i in range(len(PoseTrack_COCO_Keypoint_Ordering)):
        k = PoseTrack_COCO_Keypoint_Ordering[i]
        heatmap = heatmaps[i]
        heatmap = np.clip(heatmap * 255, 0, 255).astype(np.uint8)
        colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_BONE)
        colored_heatmap = cv2.resize(colored_heatmap, (img.shape[1], img.shape[0]))
        img_heatmap = img * 0.3 + colored_heatmap * 0.7
        cv2.imwrite(os.path.join(dir_out, '{}{}_img_heatmap.png'.format(name, k)), img_heatmap)

def dist_acc(dists, thr=0.5, percentage=True):
    ''' Return percentage below threshold while ignoring values with a -1 '''
    dist_cal = np.not_equal(dists, -1)
    num_dist_cal = dist_cal.sum()
    if num_dist_cal > 0:
        less_thr_count = np.less(dists[dist_cal], thr).sum() * 1.0
        if percentage:
            return less_thr_count / num_dist_cal
        else:
            return less_thr_count, num_dist_cal  # less_thr_count = match  / num_dist_cal （val）
    else:
        if percentage:
            return -1
        else:
            return -1, -1


def pck_accuracy(output, target, box_xywh, hm_type="gaussian", thr=0.5, ):
    '''
        In order to evaluate the effect of the model on PennAction.

        Calculate accuracy according to PCK (),
        but uses ground truth heatmap rather than x,y locations
        First value to be returned is average accuracy across 'idxs',
        followed by individual accuracies
    '''
    idx = list(range(output.shape[1]))
    norm = 1.0
    if hm_type == 'gaussian':
        pred, _ = get_max_preds(output)
        target, _ = get_max_preds(target)
        h = output.shape[2]
        w = output.shape[3]

        box_w = box_xywh[2].numpy()
        box_h = box_xywh[3].numpy()

        body_size = np.max(np.stack([box_w, box_h], -1), axis=1) / 4  # image size  -> heatmaps size
        # pck 0.2
        norm = body_size

        # norm = np.ones((pred.shape[0], 1)) * batch_body_size * aspect
        # norm = np.ones((pred.shape[0], 2)) * np.array([h, w]) / 10

    dists = calc_dists(pred, target, norm)  # use a fixed length as a measure rather than the length of body parts

    acc = np.zeros((len(idx) + 1))
    avg_acc = 0
    cnt = 0

    total_num_cnt = 0
    match_num_cnt = 0
    match_cnt = [0 for _ in range(len(idx))]
    num_cnt = [0 for _ in range(len(idx))]
    for i in range(len(idx)):
        # pck 0.2
        match_cnt_item, num_cnt_item = dist_acc(dists[idx[i]], thr, percentage=False)
        if num_cnt_item > 0:
            match_cnt[i] = int(match_cnt_item)
            num_cnt[i] = int(num_cnt_item)
            acc[i + 1] = match_cnt_item / num_cnt_item
        else:
            acc[i + 1] = -1
        # acc[i + 1] = dist_acc(dists[idx[i]], thr, percentage=False)
        if acc[i + 1] >= 0:
            avg_acc = avg_acc + acc[i + 1]
            cnt += 1

    avg_acc = avg_acc / cnt if cnt != 0 else 0
    if cnt != 0:
        acc[0] = avg_acc

    return acc, avg_acc, cnt, pred, match_cnt, num_cnt


def pck_accuracy_origin_image(pred, target, box_xywh, thr=0.5, ):
    '''
        In order to evaluate the effect of the model on PennAction.

        Calculate accuracy according to PCK (),
        but uses ground truth heatmap rather than x,y locations
        First value to be returned is average accuracy across 'idxs',
        followed by individual accuracies
    '''
    idx = list(range(pred.shape[1]))
    norm = 1.0
    # if hm_type == 'gaussian':
    #     # pred, _ = get_max_preds(output)
    #     # target, _ = get_max_preds(target)

    # h = output.shape[2]
    # w = output.shape[3]

    box_w = box_xywh[2].numpy()
    box_h = box_xywh[3].numpy()

    body_size = np.max(np.stack([box_w, box_h], -1), axis=1)  # image size  -> heatmaps size
    # pck 0.2
    norm = body_size

    # norm = np.ones((pred.shape[0], 1)) * batch_body_size * aspect
    # norm = np.ones((pred.shape[0], 2)) * np.array([h, w]) / 10

    dists = calc_dists(pred, target, norm)  # use a fixed length as a measure rather than the length of body parts

    acc = np.zeros((len(idx) + 1))
    avg_acc = 0
    cnt = 0

    total_num_cnt = 0
    match_num_cnt = 0
    match_cnt = [0 for _ in range(len(idx))]
    num_cnt = [0 for _ in range(len(idx))]
    for i in range(len(idx)):
        # pck 0.2
        match_cnt_item, total_cnt_item = dist_acc(dists[idx[i]], thr, percentage=False)
        if total_cnt_item > 0:
            match_cnt[i] = int(match_cnt_item)
            num_cnt[i] = int(total_cnt_item)
            acc[i + 1] = match_cnt_item / total_cnt_item
        else:
            acc[i + 1] = -1
        # acc[i + 1] = dist_acc(dists[idx[i]], thr, percentage=False)
        if acc[i + 1] >= 0:
            avg_acc = avg_acc + acc[i + 1]
            cnt += 1


    avg_acc = avg_acc / cnt if cnt != 0 else 0
    if cnt != 0:
        acc[0] = avg_acc

    return acc, avg_acc, cnt, pred, match_cnt, num_cnt


def accuracy(output, target, hm_type='gaussian', thr=0.5):
    '''
    Calculate accuracy according to PCK (),
    but uses ground truth heatmap rather than x,y locations
    First value to be returned is average accuracy across 'idxs',
    followed by individual accuracies
    '''
    idx = list(range(output.shape[1]))
    norm = 1.0
    if hm_type == 'gaussian':
        pred, _ = get_max_preds(output)
        target, _ = get_max_preds(target)
        h = output.shape[2]
        w = output.shape[3]
        norm = np.ones((pred.shape[0], 2)) * np.array([h, w]) / 10
    dists = calc_dists(pred, target, norm)  # use a fixed length as a measure rather than the length of body parts

    acc = np.zeros((len(idx) + 1))
    avg_acc = 0
    cnt = 0

    for i in range(len(idx)):
        acc[i + 1] = dist_acc(dists[idx[i]], thr)
        if acc[i + 1] >= 0:
            avg_acc = avg_acc + acc[i + 1]
            cnt += 1

    avg_acc = avg_acc / cnt if cnt != 0 else 0
    if cnt != 0:
        acc[0] = avg_acc

    return acc, avg_acc, cnt, pred
