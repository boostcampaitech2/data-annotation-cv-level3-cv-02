import os.path as osp
import json
import os
import copy
from tqdm import tqdm

def maybe_mkdir(x):
    if not osp.exists(x):
        os.makedirs(x)

add_data_dir = os.environ.get('SM_CHANNEL_TRAIN', '../input/data/camper')
# 여기 부분 가지고 계신 폴더구성에 맞추어 수정해주시구요

with open(osp.join(add_data_dir, 'ufo/{}.json'.format('train')), 'r') as f:
# 여기 부분도 가지고 계신 파일명에 맞추어서 수정해주시길 요청드립니다.
    anno = json.load(f)
# illegibility는 전부 단어 이므로 false
ille = False
anno = anno['images']

anno_temp = copy.deepcopy(anno)

count = 0
count_normal = 0

for img_name, img_info in tqdm(anno.items()) :
    if img_info['words'] == {} :
        del(anno_temp[img_name])
        continue
    for obj, obj_info in img_info['words'].items() :
        # illegibility는 전부 단어 이므로 false 했으나 필요없어진 코드
        anno_temp[img_name]['words'][obj]['illegibility'] = False
        LE = len(img_info['words'][obj]['points'])
        if LE == 4 :
            count_normal += 1
            continue

        elif LE < 4 : #삼각형 무시함.
            del(anno_temp[img_name]['words'][obj])

        else : # 4각형 이상은 캠퍼들의 annotation규칙 잘 지켰다고 가정하에 짝수개의 좌표니까 4개로 박스로 나눠서 illegibility 했어~
            anno_temp[img_name]['words'][obj]['illegibility'] = True
            over_polygon_temp = copy.deepcopy(anno_temp[img_name]['words'][obj])
            ill = copy.deepcopy(over_polygon_temp)
            ill['points'] = []
            for index in range(LE//2 -1):
                ill['points'].append(over_polygon_temp['points'][index])
                ill['points'].append(over_polygon_temp['points'][index+1])
                ill['points'].append(over_polygon_temp['points'][-index-1])
                ill['points'].append(over_polygon_temp['points'][-index])
                anno_temp[img_name]['words'][obj+f'{index+911}'] = copy.deepcopy(ill) #911 내 생일 ㅇ_<
                ill['points'] = []
            del anno_temp[img_name]['words'][obj]
            count += 1
            
print(f'normal polygon count : {count_normal}')
print(f'deleted {count} over polygon')

anno = {'images': anno_temp}

ufo_dir = osp.join('../input/data/camper', 'ufo')
maybe_mkdir(ufo_dir)
with open(osp.join(ufo_dir, 'train.json'), 'w') as f:
    json.dump(anno, f, indent=4)