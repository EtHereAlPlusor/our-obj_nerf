import imageio
import numpy as np

if __name__ == '__main__':
    path = '/data/datasets/KITTI-360/data_2d_semantics/train/2013_05_28_drive_0000_sync/image_00/instance/0000005308.png'
    im = imageio.imread(path)
    instances = []
    area = 0
    # print(im.shape)
    for row in range(im.shape[0]):
        for col in range(im.shape[1]):
            if im[row][col] == 26323:   # instance id
                area += 1
    area = area / (im.shape[0]*im.shape[1])
    print(area)
    #             instances.append(im[row][col])
    # instances = np.unique(instances)
    # print(instances)
                