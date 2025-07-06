import numpy as np
from .frequency_domain import (
    ideal_lowpass_filter,
    gaussian_lowpass_filter,
    butterworth_lowpass_filter
)

from .spatial_domain import (
    mean_filter,
    gaussian_filter,
    lanczos_filter,
    sinc_filter,
    kaiser_filter
)

filter_names = [
    'gaussian_lowpass', 'ideal_lowpass', 'butterworth_lowpass',
    'mean_spatial', 'gaussian_spatial', 'sinc_spatial', 'lanczos_spatial', 'kaiser_spatial'
] 

def apply_freq_filter(image, filter_name, cutoff_freq):   
    filter_func = {
        "gaussian": gaussian_lowpass_filter,
        "ideal": ideal_lowpass_filter,
        "butterworth": butterworth_lowpass_filter
    }
    assert filter_name.lower() in filter_func, f"Unknown filter name {filter_name}."
    # unfold gray image to (h,w,1)
    if image.ndim == 2:
        image = image[..., np.newaxis]
    # result mat
    new_image = np.zeros_like(image)
    h, w = image.shape[:2]
    c = image.shape[2] if len(image.shape) == 3 else 1
    # channel-wise filter
    for ch in range(c):
        channel = image[:,:,ch]
        # Fourier transform
        f_transform = np.fft.fft2(channel)
        f_shift = np.fft.fftshift(f_transform)
        # create filter and apply
        filter_mat = filter_func[filter_name](h, w, cutoff_freq)
        filtered_shift = f_shift * filter_mat
        # Inverse fourier transform
        f_ishift = np.fft.ifftshift(filtered_shift)
        channel_back = np.fft.ifft2(f_ishift)
        filtered_channel = np.abs(channel_back)
        # store result
        new_image[:, :, ch] = filtered_channel

    new_image = new_image.squeeze()
    if new_image.dtype == np.float32:
        return np.clip(new_image, 0, 1)
    else:
        return np.clip(new_image, 0, 255).astype(np.unit8)


def apply_spatial_filter(image, filter_name, kernel_size, **kwargs):
    assert filter_name in ['mean', 'gaussian', 'sinc', 'lanczos', 'kaiser']
    filter_dict = {
        'mean': mean_filter,
        'gaussian': gaussian_filter,
        'sinc': sinc_filter,
        'lanczos': lanczos_filter,
        'kaiser': kaiser_filter
    }
    return filter_dict[filter_name.lower()](image, kernel_size, **kwargs)


def filter(image, filter_name, cutoff_freq=50, kernel_size=7, **kwargs):
    if 'lowpass' in filter_name:
        name = filter_name.split('_')[0]
        return apply_freq_filter(image, name, cutoff_freq)
    elif 'spatial' in filter_name:
        name = filter_name.split('_')[0]
        return apply_spatial_filter(image, name, kernel_size)
    else:
        raise ValueError