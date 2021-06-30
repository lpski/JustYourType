import h5py, math, numpy as np, string
from PIL import Image
from typing import List, Optional


def get_data() -> np.ndarray:
    print('\nFetching Font Data')
    f = h5py.File('data/fonts.hdf5', 'r')
    return f['fonts']


def get_characters() -> List[str]:
    # TODO: Add support for 0-9
    chars = [f'{c}{c}' for c in string.ascii_uppercase] + [c for c in string.ascii_lowercase]
    return chars


def draw_grid(data: np.ndarray, cols: Optional[int]=None):
    n = data.shape[0]
    if cols is None:
        cols = int(math.ceil(n**0.5))
    rows = int(math.ceil(1.0 * n / cols))
    data = data.reshape((n, 64, 64))

    img = Image.new('L', (cols * 64, rows * 64), 255)
    for z in range(n):
        x, y = z % cols, z // cols
        img_char = Image.fromarray(np.uint8(((1.0 - data[z]) * 255)))
        img.paste(img_char, (x * 64, y * 64))

    return img

