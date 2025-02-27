import cv2
import torch
import numpy as np
from scipy.ndimage import map_coordinates, gaussian_filter

def elastic_transform(image1, image2, mask, alpha, sigma, random_state=None):
    if random_state is None:
        random_state = np.random.RandomState(None)
    
    shape = image1.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))

    image1 = map_coordinates(image1, indices, order=1, mode='reflect').reshape(shape)
    image2 = map_coordinates(image2, indices, order=1, mode='reflect').reshape(shape)
    mask = map_coordinates(mask, indices, order=1, mode='reflect').reshape(shape)

    return image1, image2, mask

class Normalize(object):
    """Normalize the image in a sample to a given intensity.

    Args:
        output_intensity (tuple or int):
    """

    def __init__(self, output_intensity=1):
        assert isinstance(output_intensity, (int, tuple))
        self.output_intensity = output_intensity

    def __call__(self, sample):
        image, masks = sample['image'], sample['mask']

        img = cv2.normalize(image, None, 0, self.output_intensity, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        msk = cv2.normalize(masks, None, 0, self.output_intensity, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        return {'image': img, 'mask': msk.astype(int)}


class NormalizeSiamese(object):
    """Normalize and transform sample

    Args:
        output_intensity (tuple or int):
    """

    def __init__(self, v_flip=False, h_flip=False, elastic_transform=False, output_intensity=1, p=0.5):
        assert isinstance(output_intensity, (int, tuple))
        self.output_intensity = output_intensity
        self.p = p
        self.v_flip = v_flip
        self.h_flip = h_flip
        self.elastic_transform = elastic_transform

    def __call__(self, sample):
        image_1, image_2, masks = sample['image_1'], sample['image_2'], sample['mask']

        img1 = cv2.normalize(image_1, None, 0, self.output_intensity, cv2.NORM_MINMAX, dtype=cv2.CV_32F)[0]
        img2 = cv2.normalize(image_2, None, 0, self.output_intensity, cv2.NORM_MINMAX, dtype=cv2.CV_32F)[0]
        msk = cv2.normalize(masks, None, 0, self.output_intensity, cv2.NORM_MINMAX, dtype=cv2.CV_32F)[0]

        if self.v_flip:
            if torch.rand(1) < self.p:
                img1 = cv2.flip(img1, 0)
                img2 = cv2.flip(img2, 0)
                msk = cv2.flip(msk, 0)
        if self.h_flip:
            if torch.rand(1) < self.p:
                img1 = cv2.flip(img1, 1)
                img2 = cv2.flip(img2, 1)
                msk = cv2.flip(msk, 1)
        if self.elastic_transform:
            if torch.rand(1) < self.p:
                img1, img2, msk = elastic_transform(img1, img2, msk, 30, 4)


        return {'image_1': np.expand_dims(img1, axis=0),
                'image_2': np.expand_dims(img2, axis=0),
                  'mask': np.expand_dims(msk.astype(int), axis=0)}

class NormalizeBASiamese(object):
    """Normalize and transform sample

    Args:
        output_intensity (tuple or int):
    """

    def __init__(self, output_intensity=1, v_flip=False, h_flip=False, elastic_transform=False, p=0.5):
        assert isinstance(output_intensity, (int, tuple))
        self.output_intensity = output_intensity
        self.p = p
        self.v_flip = v_flip
        self.h_flip = h_flip
        self.elastic_transform = elastic_transform

    def __call__(self, sample):
        image_1, image_2, masks, boundary = sample['image_1'], sample['image_2'], sample['mask'], sample['boundary']

        img1 = cv2.normalize(image_1, None, 0, self.output_intensity, cv2.NORM_MINMAX, dtype=cv2.CV_32F)[0]
        img2 = cv2.normalize(image_2, None, 0, self.output_intensity, cv2.NORM_MINMAX, dtype=cv2.CV_32F)[0]
        msk = cv2.normalize(masks, None, 0, self.output_intensity, cv2.NORM_MINMAX, dtype=cv2.CV_32F)[0]
        boundary = cv2.normalize(boundary, None, 0, self.output_intensity, cv2.NORM_MINMAX, dtype=cv2.CV_32F)[0]
        
        if self.v_flip:
            if torch.rand(1) < self.p:
                img1 = cv2.flip(img1, 0)
                img2 = cv2.flip(img2,0)
                msk = cv2.flip(msk,0)
                boundary = cv2.flip(boundary,0)
        if self.h_flip:
            if torch.rand(1) < self.p:
                img1 = cv2.flip(img1, 1)
                img2 = cv2.flip(img2,1)
                msk = cv2.flip(msk,1)
                boundary = cv2.flip(boundary,1)
        if self.elastic_transform:
            if torch.rand(1) < self.p:
                img1, img2, msk = elastic_transform(img1, img2, msk, 30, 4)

        return  {'image_1': np.expand_dims(img1, axis=0),
                 'image_2': np.expand_dims(img2, axis=0),
                 'mask': np.expand_dims(msk.astype(int), axis=0),
                 'boundary': np.expand_dims(boundary.astype(int), axis=0)}