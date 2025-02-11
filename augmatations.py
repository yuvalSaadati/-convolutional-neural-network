import numpy as np
from scipy.ndimage import rotate

np.random.seed(3)


def gaussian_noise(img, mean=0, sigma=0.03):
    img = img.copy()
    noise = np.random.normal(mean, sigma, img.shape)
    mask_overflow_upper = img + noise >= 1.0
    mask_overflow_lower = img + noise < 0
    noise[mask_overflow_upper] = 1.0
    noise[mask_overflow_lower] = 0
    img += noise
    return img


def rotate_img(img, angle, bg_patch=(5, 5)):
    assert len(img.shape) <= 3, "Incorrect image shape"
    rgb = len(img.shape) == 3
    if rgb:
        bg_color = np.mean(img[:bg_patch[0], :bg_patch[1], :], axis=(0, 1))
    else:
        bg_color = np.mean(img[:bg_patch[0], :bg_patch[1]])
    img = rotate(img, angle, reshape=False)
    mask = [img <= 0, np.any(img <= 0, axis=-1)][rgb]
    img[mask] = bg_color
    return img


def random_crop10(img, crop_size=(10, 10)):
    assert crop_size[0] <= img.shape[0] and crop_size[1] <= img.shape[1], "Crop size should be less than image size"
    img = img.copy()
    w, h = img.shape[:2]
    x, y = np.random.randint(h - crop_size[0]), np.random.randint(w - crop_size[1])
    img = img[y:y + crop_size[0], x:x + crop_size[1]]
    return img


def resize(img, target_size=(32, 32)):
    """
    Fast bilinear interpolation for resizing an image using NumPy (vectorized).

    Args:
        img (numpy.ndarray): Input image of shape (H, W, C).
        target_size (tuple): Desired output size (new_H, new_W).

    Returns:
        numpy.ndarray: Resized image of shape (new_H, new_W, C).
    """
    h, w, c = img.shape
    new_h, new_w = target_size

    # Compute scale factors
    scale_x = np.linspace(0, w - 1, new_w)
    scale_y = np.linspace(0, h - 1, new_h)

    # Compute integer coordinates
    x0 = np.floor(scale_x).astype(int)
    x1 = np.minimum(x0 + 1, w - 1)
    y0 = np.floor(scale_y).astype(int)
    y1 = np.minimum(y0 + 1, h - 1)

    # Compute interpolation weights
    wx = scale_x - x0
    wy = scale_y - y0

    # Extract pixel values using broadcasting
    Ia = img[np.ix_(y0, x0)]
    Ib = img[np.ix_(y0, x1)]
    Ic = img[np.ix_(y1, x0)]
    Id = img[np.ix_(y1, x1)]

    # Compute bilinear interpolation
    top = (1 - wx) * Ia + wx * Ib
    bottom = (1 - wx) * Ic + wx * Id
    resized = (1 - wy[:, None]) * top + wy[:, None] * bottom

    return resized.astype(img.dtype)


def random_crop28(img, crop_size=(28, 28)):
    """Randomly crops a 28x28 patch from a 32x32 image and resizes it back to 32x32."""
    h, w, c = img.shape
    ch, cw = crop_size
    x = np.random.randint(0, w - cw + 1)
    y = np.random.randint(0, h - ch + 1)
    cropped = img[y:y + ch, x:x + cw, :]
    return resize(cropped, (w, h))


def translate(img, shift=10, direction='right', roll=True):
    assert direction in ['right', 'left', 'down', 'up'], 'Directions should be top|up|left|right'
    img = img.copy()
    if direction == 'right':
        right_slice = img[:, -shift:].copy()
        img[:, shift:] = img[:, :-shift]
        if roll:
            img[:, :shift] = np.fliplr(right_slice)
    if direction == 'left':
        left_slice = img[:, :shift].copy()
        img[:, :-shift] = img[:, shift:]
        if roll:
            img[:, -shift:] = left_slice
    if direction == 'down':
        down_slice = img[-shift:, :].copy()
        img[shift:, :] = img[:-shift, :]
        if roll:
            img[:shift, :] = down_slice
    if direction == 'up':
        upper_slice = img[:shift, :].copy()
        img[:-shift, :] = img[shift:, :]
        if roll:
            img[-shift:, :] = upper_slice
    return img


def distort(img, orientation='horizontal', func=np.sin, x_scale=0.05, y_scale=5):
    assert orientation[:3] in ['hor', 'ver'], "dist_orient should be 'horizontal'|'vertical'"
    assert func in [np.sin, np.cos], "supported functions are np.sin and np.cos"
    assert 0.00 <= x_scale <= 0.1, "x_scale should be in [0.0, 0.1]"
    assert 0 <= y_scale <= min(img.shape[0], img.shape[1]), "y_scale should be less then image size"
    img_dist = img.copy()

    def shift(x):
        return int(y_scale * func(np.pi * x * x_scale))

    for c in range(3):
        for i in range(img.shape[orientation.startswith('ver')]):
            if orientation.startswith('ver'):
                img_dist[:, i, c] = np.roll(img[:, i, c], shift(i))
            else:
                img_dist[i, :, c] = np.roll(img[i, :, c], shift(i))

    return img_dist


def img_distort(img):
    imgs_distorted = []
    for ori in ['ver', 'hor']:
        for x_param in [0.01, 0.02, 0.03, 0.04]:
            for y_param in [2, 4, 6, 8, 10]:
                imgs_distorted.append(distort(img, orientation=ori, x_scale=x_param, y_scale=y_param))


def change_channel_ratio(img, channel='r', ratio=0.5):
    assert channel in 'rgb', "Value for channel: r|g|b"
    img = img.copy()
    ci = 'rgb'.index(channel)
    img[:, :, ci] *= ratio
    return img


def change_channel_ratio_gauss(img, channel='r', mean=0, sigma=0.03):
    assert channel in 'rgb', "cahenel must be r|g|b"
    img = img.copy()
    ci = 'rgb'.index(channel)
    img[:, :, ci] = gaussian_noise(img[:, :, ci], mean=mean, sigma=sigma)
    return img


def random_brightness(img, factor_range=(0.7, 1.3)):
    """Randomly adjusts brightness by multiplying pixel values by a factor."""
    factor = np.random.uniform(*factor_range)
    img = np.clip(img * factor, 0, 255).astype(np.uint8)
    return img


def add_gaussian_noise(img, mean=0, std=10):
    """Adds Gaussian noise to an image."""
    noise = np.random.normal(mean, std, img.shape).astype(np.uint8)
    noisy_img = np.clip(img + noise, 0, 255)
    return noisy_img


def augment_images(images):
    """
    Augments a batch of images to expand the dataset 3x.

    Args:
        images (numpy.ndarray): Original images of shape (N, 3, 32, 32)

    Returns:
        numpy.ndarray: Augmented images of shape (3*N, 3, 32, 32)
    """
    num_images = images.shape[0]
    augmented_images = []

    for i in range(num_images):
        img = images[i]  # Shape: (3, 32, 32)

        # Convert channel-first to channel-last (H, W, C)
        img = np.transpose(img, (1, 2, 0))

        # Original image
        augmented_images.append(img)

        # Augmentations
        flipped_horiz = np.flip(img, 1)  # Horizontal flip
        rotated = rotate(img, 90)  # Rotate 90 degrees
        bright_img = random_brightness(img)
        noisy_img = add_gaussian_noise(img)
        cropped_img = random_crop28(img)

        augmented_images.extend([flipped_horiz, rotated, bright_img, noisy_img, cropped_img, img])

    # Convert list to numpy array and back to (N, 3, 32, 32)
    augmented_images = np.array(augmented_images)
    augmented_images = np.transpose(augmented_images, (0, 3, 1, 2))  # Back to (N, 3, 32, 32)

    return augmented_images

# # Example usage
# x = np.random.randint(0, 256, (8000, 3, 32, 32), dtype=np.uint8)  # Simulated dataset
# x_augmented = augment_images(x)
#
# print("Original shape:", x.shape)  # (8000, 3, 32, 32)
# print("Augmented shape:", x_augmented.shape)
# Should be 5x the original if all augmentations are used
