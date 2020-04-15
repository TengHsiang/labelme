## Customizing labels by remapping ADE20k labels

The object information of ADE20K dataset is in [objectInfo150.txt](objectInfo150.txt).

## Usage

```bash
labelme_remap_annotations --label-file rtk --remap-table rtk --image-list training'{shift_auto_shape_color: -2}'
```

## Example of Remap Table [map_rtk.txt](map_rtk.txt)

0 # means mapping label 0 of ADE20K to label 0 of rtk

3 # means mapping label 3 of ADE20K to label 1 of rtk

13 # means mapping label 13 of ADE20K to label 2 of rtk

22, 27, 61, 105, 110, 114

44, 101, 124, 144, 145, 149

5, 18, 35, 47, 67, 73, 127

2, 9, 11, 49, 63, 85

