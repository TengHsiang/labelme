import argparse
import base64
import json
import os
import os.path as osp

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

def main():
    logger.warning('This script is aimed to convert the '
                   'JSON file to a dataset in camvid format.')

    parser = argparse.ArgumentParser()
    parser.add_argument('json_file')
    # parser.add_argument('-jl', '--json-list', default=None)
    parser.add_argument('-lf', '--label-file', default=None)
    parser.add_argument('-o', '--out', default=None)
    args = parser.parse_args()

    json_file = args.json_file
    json_list = args.json_list
    label_file = args.label_file
    
    # Import label file
    label_name_to_value = importLabel(label_file)

    # Load .json from list file
    # with open(json_list, 'r') as jl:


    if args.out is None:
        out_dir = osp.basename(json_file).replace('.', '_')
        out_dir = osp.join(osp.dirname(json_file), out_dir)
    else:
        out_dir = args.out
    if not osp.exists(out_dir):
        os.mkdir(out_dir)

    data = json.load(open(json_file))
    imageData = data.get('imageData')

    if not imageData:
        imagePath = os.path.join(os.path.dirname(json_file), data['imagePath'])
        with open(imagePath, 'rb') as f:
            imageData = f.read()
            imageData = base64.b64encode(imageData).decode('utf-8')
    img = utils.img_b64_to_arr(imageData)

    for shape in sorted(data['shapes'], key=lambda x: x['label']):
        label_name = shape['label']
        label_value = label_name_to_value[label_name]
        
    lbl, _ = utils.shapes_to_label(
        img.shape, data['shapes'], label_name_to_value
    )
    label_names = [None] * (max(label_name_to_value.values()) + 1)
    for name, value in label_name_to_value.items():
        label_names[value] = name

    lbl_viz = imgviz.label2rgb(
        label=lbl, img=imgviz.asgray(img), label_names=label_names, loc='rb'
    )

    # PIL.Image.fromarray(img).save(osp.join(out_dir, 'img.png'))
    utils.lblsave_gray(osp.join(out_dir, 'label.png'), lbl)
    # PIL.Image.fromarray(lbl_viz).save(osp.join(out_dir, 'label_viz.png'))

    with open(osp.join(out_dir, 'label_names.txt'), 'w') as f:
        for lbl_name in label_names:
            f.write(lbl_name + '\n')

    logger.info('Saved to: {}'.format(out_dir))


if __name__ == '__main__':
    main()
