import argparse
import base64
import json
import os
import os.path as osp
import numpy as np
import imgviz
import PIL.Image

from labelme.logger import logger
from labelme import utils

def importLabel(label_file):
    label_name_to_value = {}
    with open(label_file, 'r') as f:
        label_idx = 0
        while True:
            labelName = f.readline()
            if not labelName:
                break
            label_name_to_value[labelName.strip('\r\n')] = label_idx
            label_idx+=1

    return label_name_to_value

def importLabelList(image_list):
    labelList = []
    with open(image_list, 'r') as f:
        while True:
            imgPath = f.readline()
            if not imgPath:
                break
            imgPath = imgPath.strip('\r\n').replace('jpg','png')
            labelList.append(osp.join('annotations', imgPath.split('/')[0], imgPath.split('/')[1]))

    return labelList

def importRemapTable(map_table):
    mapping = []
    with open(map_table, 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break
            line = line.strip('\r\n')
            mapping.append(line)

    return mapping

def remapLabel(label_img, mapping):
    output = np.zeros(label_img.shape, dtype=int)
    for i in range(1, len(mapping)):
        map_list = list(map(int, mapping[i].split(', ')))
        for j in range(0,len(map_list)):
            # print(i, map_list, j, int(map_list[j]))
            output[label_img == int(map_list[j])] = i
    
    return output

def main():
    logger.warning('This script is aimed to remap the ADE20K '
                   'annotations to a customize annotation.')

    parser = argparse.ArgumentParser()
    parser.add_argument('--label-file', default="rtk")
    parser.add_argument('--remap-table', default="rtk")
    parser.add_argument('--image-list', default="training")
    parser.add_argument('--save-vizImage', default=False)
    parser.add_argument('--save-oriImage', default=False)
    # parser.add_argument('--save-colorLabImage', default=False)
    args = parser.parse_args()
    
    # Import label file
    label_file = 'labels_' + args.label_file + '.txt'
    label_name_to_value = importLabel(label_file)
    
    label_names = [None] * (max(label_name_to_value.values()) + 1)
    for name, value in label_name_to_value.items():
        label_names[value] = name
        # print(value, " ", name)

    # Import remap table
    map_table = 'map_' + args.remap_table + '.txt'
    mapping = importRemapTable(map_table)

    # Output direction
    out_folder = osp.join('annotations_' + args.label_file, args.image_list)
    if not osp.exists(out_folder):
        os.makedirs(out_folder, exist_ok=True)
    logger.info('Saved to: {}'.format(out_folder))

    # Load label image list
    label_list = importLabelList(args.image_list + '.txt')
    
    # Main loop
    for idx in range(0, len(label_list)):
        # load label image
        with open(label_list[idx], 'rb') as f:
            image_name = osp.split(label_list[idx])[1].split('.')[0]
            imageData = f.read()
            if not imageData:
                logger.info('Lebelled Image does not existed')
                break
            imageData = base64.b64encode(imageData).decode('utf-8')
            label_img = utils.img_b64_to_arr(imageData)
            label_img = remapLabel(label_img, mapping)

        utils.lblsave_gray(osp.join(out_folder, image_name + '.png'), label_img)

        # if args.save_colorLabImage:
        #     utils.lblsave(osp.join(out_folder, 'label_color.png'), label_img)

        # load original image
        if args.save_oriImage or args.save_vizImage:
            image_list = label_list[idx].replace('annotations','images')
            image_list = image_list.replace('png','jpg')
            with open(image_list, 'rb') as f:
                imageData = f.read()
                if not imageData:
                    logger.info('Original Color Image does not existed')
                    args.save_oriImage = args.save_vizImage = False
                imageData = base64.b64encode(imageData).decode('utf-8')
                img = utils.img_b64_to_arr(imageData)

            if args.save_oriImage:
                PIL.Image.fromarray(img).save(osp.join(out_folder, image_name+'.jpg'))
            
            if args.save_vizImage:
                lbl_viz = imgviz.label2rgb(
                    label=label_img, img=imgviz.asgray(img), label_names=label_names, loc='rb'
                )
                PIL.Image.fromarray(lbl_viz).save(osp.join(out_folder, image_name+'_viz.png'))
        
        # break

if __name__ == '__main__':
    main()

