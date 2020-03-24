import os.path as osp

import numpy as np
import PIL.Image


def lblsave(filename, lbl):
    import imgviz

    if osp.splitext(filename)[1] != '.png':
        filename += '.png'
    # Assume label ranses [-1, 254] for int32,
    # and [0, 255] for uint8 as VOC.
    if lbl.min() >= -1 and lbl.max() < 255:
        lbl_pil = PIL.Image.fromarray(lbl.astype(np.uint8), mode='P')
        colormap = imgviz.label_colormap()
        lbl_pil.putpalette(colormap.flatten())
        lbl_pil.save(filename)
    else:
        raise ValueError(
            '[%s] Cannot save the pixel-wise class label as PNG. '
            'Please consider using the .npy format.' % filename
        )

def lblsave_gray(filename, lbl):
    import imgviz
    from labelme import utils
    if osp.splitext(filename)[1] == '.png':
        filename = osp.splitext(filename)[0]

    # Assume label ranses [-1, 254] for int32,
    # and [0, 255] for uint8 as VOC.
    if lbl.min() >= -1 and lbl.max() < 255:
        lbl_pil = PIL.Image.fromarray(lbl.astype(np.uint8), mode='P')
        colormap = imgviz.label_colormap()
        graymap = utils.label_graymap()
        lbl_pil.putpalette(graymap.flatten())
        lbl_pil.save(filename+'.png')
        lbl_pil.putpalette(colormap.flatten())
        lbl_pil.save(filename+'_color.png')
    else:
        raise ValueError(
            '[%s] Cannot save the pixel-wise class label as PNG. '
            'Please consider using the .npy format.' % filename
        )