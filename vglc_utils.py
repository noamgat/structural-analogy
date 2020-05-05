import collections
import math
import cv2
import PIL
import torchvision
import numpy as np
from skimage.transform import resize
from vglc_reader import VGLCLevelRepresentationType, VGLCGameData
from typing import List

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
                # Test that the balance between channels (% ground, % enemies etc) is equal to that of the source
                channel_balance_similarity_check = scale_channels.mean(axis=(0, 1)) - image.mean(axis=(0, 1))
                # Test that every pixel sums to 1
                pixel_balance_text = scale_channels.sum(axis=2)
                res.append(scale_channels)
            levels.append(res)

    save_visualizations_on_load = True
    if save_visualizations_on_load:
        for levelidx, level in enumerate(levels):
            fn = f'VisualizeLevels/TestLevel{levelidx}.png'
            arrs = [visualize_level(game, arr) for arr in level]
            save_numpy_rgb_uint8_images(arrs, fn)
    return levels


def get_tile_color(tile_properties):
    if 'empty' in tile_properties:
        return 255, 255, 255
    if 'hazard' in tile_properties:
        return 255, 0, 0
    if 'collectable' in tile_properties:
        return 255, 255, 0
    return 150, 75, 0


def visualize_level(vglc_game_data: VGLCGameData, population_vector: np.ndarray):
    per_pixel_output_dimension = 5
    pixels_per_population = (per_pixel_output_dimension ** 2)
    stride = per_pixel_output_dimension + 1
    result = np.zeros((population_vector.shape[0]*stride, population_vector.shape[1]*stride, 3), dtype=np.uint8)
    indexed_colors = [get_tile_color(vglc_game_data.tile_info[k]) for k in vglc_game_data.sorted_tile_types]
    for y in range(population_vector.shape[0]):
        for x in range(population_vector.shape[1]):
            population = population_vector[y, x]
            histogram = collections.defaultdict(int)
            for index, density in enumerate(population):
                histogram[indexed_colors[index]] += density
            pixels_placed = 0
            for color, density in histogram.items():
                pixels_to_place = int(round(density * pixels_per_population))
                for _ in range(pixels_to_place):
                    if pixels_placed < pixels_per_population:
                        result[y*stride + (pixels_placed // per_pixel_output_dimension), x*stride + (pixels_placed % per_pixel_output_dimension)] = color
                        pixels_placed += 1
    return result


def save_numpy_rgb_uint8_images(arr: List[np.ndarray], filename: str):
    from PIL import Image
    images = [Image.fromarray(a) for a in arr]
    widths, heights = zip(*(i.size for i in images))

    max_width = max(widths)
    total_height = sum(heights)

    new_im = Image.new('RGB', (max_width, total_height))

    y_offset = 0
    for im in images:
        new_im.paste(im, (0, y_offset))
        y_offset += im.size[1]

    new_im.save(filename)