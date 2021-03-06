import sys
import numpy as np, pandas as pd, os, shutil
from typing import Dict, List, Optional, Set, Tuple
from imageio import imread
from random import sample
from utils import get_characters
from PIL import Image, ImageDraw, ImageChops
from sklearn.preprocessing import LabelEncoder


def get_available_fonts() -> List[str]:
    fonts = pd.read_csv('data/primary/fonts/all.csv')
    ignore = set(pd.read_csv('data/primary/fonts/ignore.csv')['name'].values)
    available = [f.strip() for f in fonts['name'].values]
    return [f for f in available if f not in ignore]



# Fetching
def fetch_fonts():
    # https://static-webfonts.myfonts.com/kit/RooneySans-Regular_bold_normal/font.eot
    # https://static-webfonts.myfonts.com/set-sail-studios/madina-script/font.woff
    pass



# Clean up / Set up
def extract_data():
    fonts = get_available_fonts()
    print(f'{len(fonts)} fonts available: {fonts[:10]}')


def clean_folder():
    fonts = get_available_fonts()
    chars = get_characters()
    for font in fonts:
        for c in chars:
            from_folder = 'data/primary/fontimage'
            to_folder = f'data/primary/images/{c}'
            if not os.path.exists(to_folder): os.mkdir(to_folder)

            char_file = f'{font}_{c}.png'
            shutil.copyfile(f'{from_folder}/{char_file}', f'{to_folder}/{char_file}')
            

def analyze_fonts(start: int, end: Optional[int] = None):
    fonts = get_available_fonts()
    if end is None: end = len(fonts)
    fonts = fonts[start:end]
    target_w, target_h = 64, 64
    chars = get_characters()

    label_df = pd.read_csv(f'data/primary/labels/labels.csv', index_col='font')
    char_encodings = { char: i for i, char in enumerate(get_characters()) }

    char_to_X: Dict[str, list] = { c: [] for c in chars }
    char_to_y: Dict[str, list] = { c: [] for c in chars }
    for i, font in enumerate(fonts):
        if i % 50 == 0: print(f'{i}/{len(fonts)} Complete')

        min_ly, max_hy, max_width, max_height = float('inf'), float('-inf'), 0, None
        imgs: List[Tuple[int, int, int, Image.Image, str]] = []

        for c in get_characters():
            try:
                # Fetch character image
                path = f'data/primary/images/{c}/{font}_{c}.png'
                pil_im = Image.open(path, 'r').convert('L')
                
                # Get true width
                blank = Image.new('L', (720, 720), 255)
                diff = ImageChops.difference(pil_im, blank)

                lx, ly, hx, hy = diff.getbbox()
                min_ly = min(min_ly, ly)
                max_hy = max(max_hy, hy)
                max_width = max(max_width, hx - lx)
                
                if max_height is None or max_height[0] < hy - ly:
                    max_height = (hy-ly, c)
                imgs.append((lx, hx, hy-ly, pil_im, c))

            except Exception as e:
                print(f'Error processing {font}:{c} | {e}')
                add_font_to_ignored(font)
                imgs = []
                break
            
        
        # Scale all characters in font
        try: scale_factor = min(1.0 * target_h / (max_hy - min_ly), 1.0 * target_w / max_width)
        except Exception as e: 
            add_font_to_ignored(font)
            continue

        # Get one hot encoding for font + char label
        try:
            tags = np.array(label_df.loc[font]['labels'].split(','))
        except Exception as e:
            print(f'error getting tags for {font}:', label_df.loc[font]['labels'])
            print(font in label_df.index.values)
            sys.exit(0)
            # break
        font_label = one_hot_tags(tags)

        for lx, hx, y, img, c in imgs:
            try:
                folder = f'data/primary/adj_images/{c}'
                img = img.crop((lx, min_ly, hx, max_hy))

                # Resize to smaller
                new_width = (hx-lx) * scale_factor
                new_height = (max_hy - min_ly) * scale_factor
                og_scaled_height = y * scale_factor
                img = img.resize((int(new_width), int(new_height)), Image.ANTIALIAS)
                img = img.crop((0, 0, img.size[0], og_scaled_height))

                # Expand to square
                img_sq = Image.new('L', (target_w, target_h), 255)
                offset_x = (target_w - new_width) / 2
                offset_y = (target_h - new_height) / 2
                img_sq.paste(img, (int(offset_x), int(offset_y)))
                
                # Save adjusted image
                if not os.path.exists(folder): os.mkdir(folder)
                img_sq.save(f'{folder}/{font}_{c}.png')

                # Save numpy array conversion of image
                im_data = np.array(img_sq)
                char_to_X[c].append(im_data)
                label = np.array(font_label + [char_encodings[c]])
                char_to_y[c].append(label)


                # Convert to numpy array
                # matrix = np.array(img_sq.getdata()).reshape((target_h, target_w))
                # matrix = 255 - matrix
                # data.append(matrix)
                # print(matrix[32])

            except Exception as e:
                print(f'Error post-processing {font}:{c} | {e}')
                add_font_to_ignored(font)
                break
    
        
        if i % 250 == 0: # Periodically save numpy array
            for c in chars:
                with open(f'data/primary/encoded/X_{c}.npy', 'wb') as f:
                    np.save(f, np.array(char_to_X[c]))

                with open(f'data/primary/encoded/y_{c}.npy', 'wb') as f:
                    np.save(f, np.array(char_to_y[c]))


    for c in chars:
        with open(f'data/primary/encoded/X_{c}.npy', 'wb') as f:
            np.save(f, np.array(char_to_X[c]))

        with open(f'data/primary/encoded/y_{c}.npy', 'wb') as f:
            np.save(f, np.array(char_to_y[c]))



def add_font_to_ignored(font: str):
    ignored = pd.read_csv('data/primary/fonts/ignore.csv')
    ignored = ignored.append([{ 'name': font }])
    ignored.to_csv('data/primary/fonts/ignore.csv', index=False)


def remove_symbol_fonts():
    fonts = get_available_fonts()
    ignored = pd.read_csv('data/primary/fonts/ignore.csv')

    for i, font in enumerate(fonts):
        try:
            with open(f'data/primary/labels/{font}', 'r') as f:
                labels = [l for l in f.read().strip().rstrip().split(' ') if len(l) > 0]
                if font in ignored['name'].values: continue
                elif 'non-alphabetic' in labels:
                    ignored = ignored.append([{ 'name': font }])
                elif 'dingbat' in labels:
                    ignored = ignored.append([{ 'name': font }])

        except Exception as e:
            print(f'Error processing {font}: {e}')
            if font in ignored['name'].values: continue
            else: ignored = ignored.append([{ 'name': font }])

    ignored.to_csv('data/primary/fonts/ignore.csv', index=False)


# Label Analysis
def clean_tags():
    labels: Set[str] = set()
    # label_df = pd.read_csv(f'data/primary/labels/labels.csv', index_col='font')
    label_df = pd.read_csv(f'data/primary/labels/labels.csv', index_col='font')
    label_df.info()
    
    for font in get_available_fonts():
        with open(f'data/primary/font-labels/{font}', 'r') as f:
            font_labels = [l for l in f.read().strip().rstrip().split(' ') if len(l) > 0]
            for l in font_labels: labels.add(l)
            label_df.loc[font] = [','.join(labels)]

    label_df.to_csv(f'data/primary/labels/labels.csv')
            

    # with open(f'data/primary/labels/all_labels', 'w') as f:
    #     f.write(' '.join(labels))

def tag_cardinality():
    # label_file = open(f'data/primary/labels/all_labels')
    # all_labels = [l for l in label_file.read().strip().rstrip().split(' ')]
    # label_index = { label: i for i, label in enumerate(all_labels) }

    label_df = pd.read_csv(f'data/primary/labels/labels.csv', index_col='font')
    ignore_labels = set(open('data/primary/labels/ignore_labels').read().split(','))

    counts: Dict[str, int] = {}
    for font in get_available_fonts():
        # font_labels = np.array(label_df.loc[font]['labels'].split(','))
        label_file = open(f'data/primary/font-labels/{font}')
        font_labels = label_file.read().strip().rstrip().split(' ')
        font_labels = [l for l in font_labels if l not in ignore_labels and len(l) > 0]
        if len(font_labels) == 0:
            add_font_to_ignored(font)
        else:
            label_df.loc[font] = ','.join(font_labels)
            for label in font_labels:
                counts[label] = counts.get(label, 0) + 1

    sorted_labels = sorted(counts.items(), key=lambda x: x[1])
    label_df.to_csv('data/primary/labels/labels.csv')

    print(f'{len(counts)} Unique Fonts')
    print('\n10 Most Common Labels:')
    for label, count in sorted_labels[-10:]:
        print(f'{label}: {count}')

    print('\n10 Least Common Labels:')
    for label, count in sorted_labels[:10]:
        print(f'{label}: {count}')




# Data Structuring
def one_hot_tags(labels: np.ndarray) -> List[int]:
    """Returns one hot encoding of provided labels"""
    label_file = open(f'data/primary/labels/all_labels')
    all_labels = [l for l in label_file.read().strip().rstrip().split(' ')]
    label_index = { label: i for i, label in enumerate(all_labels) }

    y = [0] * len(label_index)
    for l in labels: y[label_index[l]] = 1
    return y


def structure_data():
    fonts = get_available_fonts()[:10]
    label_df = pd.read_csv(f'data/primary/labels/labels.csv', index_col='font')
    label_file = open(f'data/primary/labels/all_labels')
    all_labels = [l for l in label_file.read().strip().rstrip().split(' ')]
    label_index = {label: i for i, label in enumerate(all_labels)}


    X, y = [], [[0] * len(label_index)] * len(fonts)

    for font in fonts:
        labels = np.array(label_df.loc[font]['labels'].split(','))
        print(type(label_df.loc[font]['labels']))
        print(len(label_df.loc[font]['labels']))

        print(f'{font}:{one_hot_tags(np.array(labels))}')


def form_encoded_data():
    dataset = Dataset()
    for char in get_characters():
    # for char in ['AA']:
        print(f'\n\n\nform_encoded_data: Processing {char}')
        X, y = dataset.encoded_fonts(char, 5000)

        with open(f'data/primary/encoded/X_{char}.npy', 'wb') as f:
             np.save(f, X)

        with open(f'data/primary/encoded/y_{char}.npy', 'wb') as f:
             np.save(f, y)


class Dataset():

    def all_fonts(self) -> List[str]:
        """Returns list of strings representing all valid fonts"""
        return get_available_fonts()

    def all_labels(self) -> List[str]:
        """Returns list of strings representing all valid labels"""
        with open(f'data/primary/labels/all_labels') as f:
            return [l for l in f.read().strip().rstrip().split(' ')]

    def font_labels(self, font: str) -> np.ndarray:
        """Returns list of labels for a given font"""
        label_df = pd.read_csv(f'data/primary/labels/labels.csv', index_col='font')
        labels = np.array(label_df.loc[font]['labels'].split(','))
        return labels

    def encode_labels(self, labels: np.ndarray) -> List[int]:
        """Returns one hot encoding of provided labels"""
        label_index = {label: i for i, label in enumerate(self.all_labels())}

        y = [0] * len(label_index)
        for l in labels: y[label_index[l]] = 1
        return y
        
    def _label_indices(self) -> Dict[str, int]:
        """Returns dictionary mapping label strings to their one hot encoded index"""
        return { label: i for i, label in enumerate(self.all_labels()) }

    def _encoded_characters(self) -> Dict[str, int]:
        """Returns dictionary mapping characters to their encoded value"""
        return { char: i for i, char in enumerate(get_characters()) }

    def _font_indices(self) -> Dict[str, int]:
        """Returns dictionary mapping font string to its encoded label"""
        return { font: i for i, font in enumerate(self.all_fonts()) }

    def _image_for_font_character(self, font: str, char: str) -> np.ndarray:
        im = Image.open(f'data/primary/adj_images/{char}/{font}_{char}.png')
        return np.array(im)

    def encoded_fonts(self, character: Optional[str] = None, count: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns encoded representation for each. If a character is provided,
            only returns representations for that character across all fonts
        """

        # font_encodings = self._font_indices()
        char_encodings = self._encoded_characters()
        fonts = self.all_fonts()
        
        X, y = [], []
        if character is None:
            for font in fonts:
                labels = self.font_labels(font)
                encoded_labels = self.encode_labels(labels)

                for char in get_characters():
                    X.append(self._image_for_font_character(font, char))
                    y.append(np.array(encoded_labels + [char_encodings[char]]))

        else:
            # Return pre-generated data if exists
            encoding_path = 'data/primary/encoded'
            if os.path.exists(f'{encoding_path}/X_{character}.npy') and os.path.exists(f'{encoding_path}/y_{character}.npy'):
                X = np.load(f'{encoding_path}/X_{character}.npy')
                y = np.load(f'{encoding_path}/y_{character}.npy')
                return X, y

            else:
                if count is None: count = len(fonts)
                for i, font in enumerate(fonts[:count]):
                    if i % 50 == 0: print(f'encoded_fonts: {i}/{count}')
                    # Gather X data
                    encoded_char = self._image_for_font_character(font, character)
                    X.append(encoded_char)

                    # Gather y data
                    labels = self.font_labels(font)
                    encoded_labels = self.encode_labels(labels)
                    y.append(np.array(encoded_labels + [char_encodings[character]]))

        X = np.array(X)
        y = np.array(y)

        return X, y

    def formatted_labels(self):
        encoding_path = 'data/primary/encoded'
        y: np.ndarray = np.load(f'{encoding_path}/y_AA.npy')

        print(y.shape)




if __name__ == '__main__':
    # extract_data()
    # clean_folder()
    analyze_fonts(0)
    # clean_tags()
    # remove_symbol_fonts()
    # tag_cardinality() # 1819 fonts + 1 character encoding = 1820 label size

    # structure_data()

    # form_encoded_data()

    # dataset = Dataset()
    # dataset.encoded_fonts('a')
    # dataset.formatted_labels()
