import math
import cv2
import PIL
import torchvision
import numpy as np
from skimage.transform import resize
from vglc_reader import VGLCLevelRepresentationType, VGLCGameData


def create_downsample_path(in_height, in_width, kernel_size, num_halfings):
    # We want the result to be more or less square. Therefore we will have more X halfings than Y
    aspect_ratio = in_width / in_height
    jump_ratio = int(round(math.log2(aspect_ratio)))
    strides = []
    paddings = []


def transform_input_vglc(json_path, opt):
    #  TODO: Extract representation type from opt
    representation_type = VGLCLevelRepresentationType.ONE_HOT
    levels = []
    game = VGLCGameData(json_path)
    for levelidx in range(len(game)):

        image = game.get_level(levelidx, representation_type).astype(float)

        def downscale_image(source_image_2d):
            h, w = source_image_2d.shape
            res = []
            for ii in range(0, opt.stop_scale + 1, 1):
                scale = math.pow(opt.scale_factor, opt.stop_scale - ii)
                # Tested PIL, numpy and CV2 resizing. CV2 was only one to preserve
                #cv2 wants w,h format even if matrix is h,w
                target_size = np.uint32(np.ceil((w * scale, h * scale)))
                sample = cv2.resize(source_image_2d, tuple(target_size), interpolation=cv2.INTER_AREA)
                res.append(sample)
            return res

        img_size = image.shape
        if len(img_size) == 2:
            levels.append(downscale_image(image))
        elif len(img_size) == 3:
            num_channels = img_size[2]
            channel_images = [image[:, :, i] for i in range(num_channels)]
            channel_scaledowns = [downscale_image(channel_image) for channel_image in channel_images]
            num_scales = len(channel_scaledowns[0])
            res = []
            for scale in range(num_scales):
                scale_channels = [channel_scaledown[scale] for channel_scaledown in channel_scaledowns]
                scale_channels = np.array(scale_channels)
                scale_channels = np.moveaxis(scale_channels, [0, 1, 2], [2, 0, 1])
                test_similarity = scale_channels.mean(axis=(0, 1)) - image.mean(axis=(0, 1))
                res.append(scale_channels)
            levels.append(res)

    return levels