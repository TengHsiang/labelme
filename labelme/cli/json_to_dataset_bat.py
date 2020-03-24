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
                   'JSON batch to gray map of DABNet format.')

    parser = argparse.ArgumentParser()
    parser.add_argument('--json-list', default=None)
    parser.add_argument('--label-file', default=None)
    args = parser.parse_args()

    # Load .json from list file
    if not osp.isfile(args.json_list):
        print("json_list doesn't existed!!")
        return
    
    with open(args.json_list, 'r') as f:
        json_files = f.readlines()
    json_files = [x.strip() for x in json_files]
    
    # Import label file
    if not osp.isfile(args.label_file):
        print("label_file doesn't existed!!")
        return
    label_name_to_value = importLabel(args.label_file)

    # main loop
    for i in range(0, len(json_files)):
        json_file = ''.join(json_files[i])
        
        out_dir = json_file.split('.')[0]
        
        data = json.load(open(json_file))
        imageData = data.get('imageData')

        if not imageData:
            imagePath = os.path.join(os.path.dirname(json_file), data['imagePath'])
            with open(imagePath, 'rb') as f:
                imageData = f.read()
                imageData = base64.b64encode(imageData).decode('utf-8')
        img = utils.img_b64_to_arr(imageData)

        # check label in json is in the label file or not
        for shape in sorted(data['shapes'], key=lambda x: x['label']):
            label_name = shape['label']
            if label_name in label_name_to_value:
                label_value = label_name_to_value[label_name]
            else:
                label_value = len(label_name_to_value)
                label_name_to_value[label_name] = label_value
                print(label_name, " is not in the label file")
        
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
        utils.lblsave_gray(out_dir + '.png', lbl)
        PIL.Image.fromarray(lbl_viz).save(out_dir + '_viz.png')

        logger.info('Saved to: {}'.format(out_dir))
        # return

if __name__ == '__main__':
    main()
