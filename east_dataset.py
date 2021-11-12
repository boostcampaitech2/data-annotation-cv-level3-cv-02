import math

import torch
import numpy as np
import cv2
from torch.utils.data import Dataset

# bounding box shrinking으로 center region 구함
def shrink_bbox(bbox, coef=0.3, inplace=False):
    lens = [np.linalg.norm(bbox[i] - bbox[(i + 1) % 4], ord=2) for i in range(4)]
    r = [min(lens[(i - 1) % 4], lens[i]) for i in range(4)]

    if not inplace:
        bbox = bbox.copy()

    offset = 0 if lens[0] + lens[2] > lens[1] + lens[3] else 1
    for idx in [0, 2, 1, 3]:
        p1_idx, p2_idx = (idx + offset) % 4, (idx + 1 + offset) % 4
        p1p2 = bbox[p2_idx] - bbox[p1_idx]
        dist = np.linalg.norm(p1p2)
        if dist <= 1:
            continue
        bbox[p1_idx] += p1p2 / dist * r[p1_idx] * coef
        bbox[p2_idx] -= p1p2 / dist * r[p2_idx] * coef
    return bbox

# bbox의 각도만큼 회전된 축 상에서 두축을 따라 선형적으로 증가하는 map을 각각 생성
# 1. x,y축을 따라 linear하게 증가하는 map 생성
# 2. find_min_rect_angle에서 구한 각도만큼 회전
def get_rotated_coords(h, w, theta, anchor):
    anchor = anchor.reshape(2, 1)
    rotate_mat = get_rotate_mat(theta)
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    x_lin = x.reshape((1, x.size))
    y_lin = y.reshape((1, x.size))
    coord_mat = np.concatenate((x_lin, y_lin), 0)
    rotated_coord = np.dot(rotate_mat, coord_mat - anchor) + anchor
    rotated_x = rotated_coord[0, :].reshape(x.shape)
    rotated_y = rotated_coord[1, :].reshape(y.shape)
    return rotated_x, rotated_y

# bounding box의 각도 구함
def get_rotate_mat(theta):
    return np.array([[math.cos(theta), -math.sin(theta)],
                     [math.sin(theta), math.cos(theta)]])

# 회전된 박스와 원본박스의 point by point 차이가 가장 적은것을 선택하도록 함 
def calc_error_from_rect(bbox):
    '''
    Calculate the difference between the vertices orientation and default orientation. Default
    orientation is x1y1 : left-top, x2y2 : right-top, x3y3 : right-bot, x4y4 : left-bot
    '''
    x_min, y_min = np.min(bbox, axis=0)
    x_max, y_max = np.max(bbox, axis=0)
    rect = np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]],
                    dtype=np.float32)
    return np.linalg.norm(bbox - rect, axis=0).sum()

# bounding box를 theta만큼 회전
def rotate_bbox(bbox, theta, anchor=None):
    points = bbox.T
    if anchor is None:
        anchor = points[:, :1]
    rotated_points = np.dot(get_rotate_mat(theta), points - anchor) + anchor
    return rotated_points.T

# bounding box의 각도 구함 1
# 몇도로 돌려야 min area rect 가 가장 작은 넓이가 되는지 각도 리턴 
def find_min_rect_angle(bbox, rank_num=10):
    '''Find the best angle to rotate poly and obtain min rectangle
    '''
    areas = []
    angles = np.arange(-90, 90) / 180 * math.pi
    for theta in angles:
        rotated_bbox = rotate_bbox(bbox, theta) # theta만큼 돌려보기
        x_min, y_min = np.min(rotated_bbox, axis=0)
        x_max, y_max = np.max(rotated_bbox, axis=0)
        areas.append((x_max - x_min) * (y_max - y_min)) # aligned bbox의 넓이(aabb)

    best_angle, min_error = -1, float('inf')
    for idx in np.argsort(areas)[:rank_num]: # 영역이 작은 순서대로 sorting
        rotated_bbox = rotate_bbox(bbox, angles[idx])
        error = calc_error_from_rect(rotated_bbox)
        if error < min_error:
            best_angle, min_error = angles[idx], error

    return best_angle


def generate_score_geo_maps(image, word_bboxes, map_scale=0.25):
    img_h, img_w = image.shape[:2]
    map_h, map_w = int(img_h * map_scale), int(img_w * map_scale) # 이미지 크기의 1/4만큼
    inv_scale = int(1 / map_scale)

    score_map = np.zeros((map_h, map_w, 1), np.float32) # 중심인지 아닌지 binary로 표시하므로 채널 1
    geo_map = np.zeros((map_h, map_w, 5), np.float32) # 각 변과 각도 정보이므로 채널 5

    word_polys = []

    # center region에 대해 score map, geo_map 생성
    for bbox in word_bboxes:
        poly = np.around(map_scale * shrink_bbox(bbox)).astype(np.int32)
        word_polys.append(poly)

        center_mask = np.zeros((map_h, map_w), np.float32)
        cv2.fillPoly(center_mask, [poly], 1) # 0으로 초기화된 score map에 [poly]영역에는 1을 채워넣어라 

        theta = find_min_rect_angle(bbox)
        rotated_bbox = rotate_bbox(bbox, theta) * map_scale
        x_min, y_min = np.min(rotated_bbox, axis=0)
        x_max, y_max = np.max(rotated_bbox, axis=0)

        anchor = bbox[0] * map_scale
        rotated_x, rotated_y = get_rotated_coords(map_h, map_w, theta, anchor)

        # geo map구성하는 d1,d2,d3,d4,theta 생성
        # 2D ReLU
        d1, d2 = rotated_y - y_min, y_max - rotated_y 
        d1[d1 < 0] = 0
        d2[d2 < 0] = 0
        d3, d4 = rotated_x - x_min, x_max - rotated_x
        d3[d3 < 0] = 0
        d4[d4 < 0] = 0
        # center_mask와 곱해서 중심 영역만 그라데이션을 갖는 값으로 만듬 
        geo_map[:, :, 0] += d1 * center_mask * inv_scale
        geo_map[:, :, 1] += d2 * center_mask * inv_scale
        geo_map[:, :, 2] += d3 * center_mask * inv_scale
        geo_map[:, :, 3] += d4 * center_mask * inv_scale
        geo_map[:, :, 4] += theta * center_mask

    cv2.fillPoly(score_map, word_polys, 1)

    return score_map, geo_map


class EASTDataset(Dataset):
    def __init__(self, dataset, map_scale=0.25, to_tensor=True):
        self.dataset = dataset
        self.map_scale = map_scale
        self.to_tensor = to_tensor

    def __getitem__(self, idx):
        image, word_bboxes, roi_mask = self.dataset[idx]
        score_map, geo_map = generate_score_geo_maps(image, word_bboxes, map_scale=self.map_scale)

        mask_size = int(image.shape[0] * self.map_scale), int(image.shape[1] * self.map_scale)
        roi_mask = cv2.resize(roi_mask, dsize=mask_size)
        if roi_mask.ndim == 2:
            roi_mask = np.expand_dims(roi_mask, axis=2)

        if self.to_tensor:
            image = torch.Tensor(image).permute(2, 0, 1)
            score_map = torch.Tensor(score_map).permute(2, 0, 1)
            geo_map = torch.Tensor(geo_map).permute(2, 0, 1)
            roi_mask = torch.Tensor(roi_mask).permute(2, 0, 1)

        return image, score_map, geo_map, roi_mask

    def __len__(self):
        return len(self.dataset)
