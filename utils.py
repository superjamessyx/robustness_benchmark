import torch
import numpy as np
import os
import torch.utils.data as data
from PIL import Image
from io import BytesIO
import cv2
from wand.image import Image as WandImage
from wand.api import library as wandlibrary
from scipy.ndimage import zoom as scizoom
from skimage import color


def metrics(prediction, target):

    prediction_binary = torch.argmax(prediction, dim=1)
    N = target.numel()

    # True positives, true negative, false positives, false negatives calculation
    tp = torch.nonzero(prediction_binary * target).shape[0]
    tn = torch.nonzero((1 - prediction_binary) * (1 - target)).shape[0]
    fp = torch.nonzero(prediction_binary * (1 - target)).shape[0]
    fn = torch.nonzero((1 - prediction_binary) * target).shape[0]

    # Metrics
    accuracy = (tp + tn) / N
    precision = (tp + 1e-4) / (tp + fp + 1e-4)
    recall = (tp + 1e-4) / (tp + fn + 1e-4)
    specificity = (tn + 1e-4) / (tn + fp + 1e-4)
    f1 = (2 * precision * recall + 1e-4) / (precision + recall + 1e-4)

    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1, 'specificity': specificity}


def clipped_zoom(img, zoom_factor):
    h = img.shape[0]
    # ceil crop height(= crop width)
    ch = int(np.ceil(h / zoom_factor))

    top = (h - ch) // 2
    img = scizoom(img[top:top + ch, top:top + ch], (zoom_factor, zoom_factor, 1), order=1)
    # trim off any extra pixels
    trim_top = (img.shape[0] - h) // 2

    return img[trim_top:trim_top + h, trim_top:trim_top + h]


def disk(radius, alias_blur=0.1, dtype=np.float32):
    if radius <= 8:
        L = np.arange(-8, 8 + 1)
        ksize = (3, 3)
    else:
        L = np.arange(-radius, radius + 1)
        ksize = (5, 5)
    X, Y = np.meshgrid(L, L)
    aliased_disk = np.array((X ** 2 + Y ** 2) <= radius ** 2, dtype=dtype)
    aliased_disk /= np.sum(aliased_disk)

    # supersample disk to antialias
    return cv2.GaussianBlur(aliased_disk, ksize=ksize, sigmaX=alias_blur)


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)

def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

# Extend wand.image.Image class to include method signature
class MotionImage(WandImage):
    def motion_blur(self, radius=0.0, sigma=0.0, angle=0.0):
        wandlibrary.MagickMotionBlurImage(self.wand, radius, sigma, angle)


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

def make_dataset(dir, class_to_idx, ratio=1.0):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            leng = int(len(fnames) * ratio)
            for fname in sorted(fnames)[:leng]:
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']

def is_image_file(filename):
    """Checks if a file is an image.
    Args:
        filename (string): path to a file
    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)




class DistortImageFolder(data.Dataset):
    def __init__(self, root, method, severity, transform=None, target_transform=None,
                 loader=default_loader):
        classes, class_to_idx = find_classes(root)
        imgs = make_dataset(root, class_to_idx)
        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                                                                             "Supported image extensions are: " + ",".join(
                IMG_EXTENSIONS)))

        self.root = root
        self.distor = Distortions()
        self.method = getattr(self.distor, method)
        self.severity = severity
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.idx_to_class = {v: k for k, v in class_to_idx.items()}
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        img = self.method(img, self.severity)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)


class Distortions(object):
    def __init__(self, ):
        self.blots = np.load('blots.npy', allow_pickle=True)
        self.bubbles = np.load('bubbles.npy', allow_pickle=True)

    def pixelate(self, x, severity=1):
        c = [0.6, 0.5, 0.4, 0.3, 0.25][severity - 1]
        dims = x.size

        x = x.resize((int(dims[0] * c), int(dims[1] * c)), Image.BOX)
        x = x.resize(dims, Image.BOX)

        return x

    def jpeg_compression(self, x, severity=1):
        c = [25, 18, 15, 10, 7][severity - 1]

        output = BytesIO()
        x.save(output, 'JPEG', quality=c)
        x = Image.open(output)

        return x

    def marking_blur(self, x, severity=1):
        x = np.array(x)
        blot = dict(self.blots[0]['96'][severity-1])
        blur = blot["blur"]
        binary = blot["mask"]
        rand_x, rand_y = blot["positions"][0]
        blur_h = blot["blur_h"]
        blur_w = blot["blur_w"]
        blur = x[rand_y: rand_y + blur_h, rand_x:rand_x + blur_w] * (1 - binary) + blur
        x[rand_y: rand_y + blur_h, rand_x:rand_x + blur_w] = blur

        return x

    def bubble_blur(self, x, severity=1):

        ratios = [0.1, 0.3, 0.5, 0.7, 0.9]
        ratio = ratios[severity-1]
        x = np.array(x)
        sample = dict(self.bubbles[0]['96'])
        bubble = sample["bubble"]
        mask = sample["mask"]
        x = x * mask * (1-ratio) + bubble * mask * ratio + x * (1 - mask)

        return np.clip(x, 0, 255).astype(np.uint8)

    def defocus_blur(self, x, severity=1):

        if x.size == (224, 224):
            c = [(3, 0.1), (4, 0.5), (6, 0.5), (8, 0.5), (10, 0.5)][severity - 1]
        elif x.size == (96, 96):
            c = [(3, 0.05), (4, 0.3), (5, 0.3), (6, 0.3), (7, 0.3)][severity - 1]

        x = np.array(x) / 255.
        kernel = disk(radius=c[0], alias_blur=c[1])

        channels = []
        for d in range(3):
            channels.append(cv2.filter2D(x[:, :, d], -1, kernel))
        channels = np.array(channels).transpose((1, 2, 0))  # 3x224x224 -> 224x224x3

        return (np.clip(channels, 0, 1) * 255).astype(np.uint8)

    def motion_blur(self, x, severity=1):
        dims = x.size
        if x.size == (224, 224):
            c = [(10, 3), (15, 5), (15, 8), (15, 12), (20, 15)][severity - 1]
        elif x.size == (96, 96):
            c = [(5, 1), (5, 2), (5, 3), (5, 4), (6, 5)][severity - 1]

        output = BytesIO()
        x.save(output, format='PNG')
        x = MotionImage(blob=output.getvalue())

        x.motion_blur(radius=c[0], sigma=c[1], angle=np.random.uniform(-45, 45))

        x = cv2.imdecode(np.fromstring(x.make_blob(), np.uint8),
                         cv2.IMREAD_UNCHANGED)

        if x.shape != dims:
            return np.clip(x[..., [2, 1, 0]], 0, 255)  # BGR to RGB
        else:  # greyscale to RGB
            return np.clip(np.array([x, x, x]).transpose((1, 2, 0)), 0, 255)

    def zoom_blur(self, x, severity=1):
        c = [np.arange(1, 1.11, 0.01),
             np.arange(1, 1.16, 0.01),
             np.arange(1, 1.21, 0.02),
             np.arange(1, 1.26, 0.02),
             np.arange(1, 1.31, 0.03)][severity - 1]

        x = (np.array(x) / 255.).astype(np.float32)
        out = np.zeros_like(x)
        for zoom_factor in c:
            out += clipped_zoom(x, zoom_factor)

        x = (x + out) / (len(c) + 1)
        return (np.clip(x, 0, 1) * 255).astype(np.uint8)

    def brightness(self, x, severity=1):
        c = [.05, .1, .15, .2, .25][severity - 1]

        x = np.array(x) / 255.
        x = color.rgb2hsv(x)
        x[:, :, 2] = np.clip(x[:, :, 2] + c, 0, 1)
        x = color.hsv2rgb(x)

        return (np.clip(x, 0, 1) * 255).astype(np.uint8)

    def saturate(self, x, severity=1):
        c = [.05, .1, .15, .2, .25][severity - 1]

        x = np.array(x) / 255.
        x = color.rgb2hsv(x)
        x[:, :, 1] = np.clip(x[:, :, 1] + c, 0, 1)
        x = color.hsv2rgb(x)

        return (np.clip(x, 0, 1) * 255).astype(np.uint8)

    def hue(self, x, severity=1):
        c = [.05, .1, .15, .2, .25][severity - 1]

        x = np.array(x) / 255.
        x = color.rgb2hsv(x)
        x[:,:,0] = np.where(x[:, :, 0] + c > 1.0, x[:, :, 0] + c -1, x[:, :, 0] + c)
        x = color.hsv2rgb(x)

        return (np.clip(x, 0, 1) * 255).astype(np.uint8)



def print_log(print_string, log):
    print("{}".format(print_string))
    log.write('{}\n'.format(print_string))
    log.flush()



def sort_metric(conf_array):
    (num, c) = conf_array.shape
    total = c * (c-1) / 2
    avg_change = 0
    for i in range(num):
        swapped = True
        cur_conf = conf_array[i]
        changes = 0
        last = c
        while swapped:
            swapped = False
            for j in range(1, last):
                if cur_conf[j - 1] > cur_conf[j]:
                    cur_conf[j], cur_conf[j - 1] = cur_conf[j - 1], cur_conf[j]  # Swap
                    changes += 1
                    swapped = True
                    last = j
        avg_change = (avg_change * i + 1.0 - changes / total) / (i+1)
    return avg_change

if __name__ == '__main__':
    result = np.load("tct_results_trans.npy", allow_pickle=True).item()
    print(result)
    # dataset = DistortImageFolder(root='/data5/zyl/patch_camelyon/test/', method='motion_blur', severity=1)
    # data_loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=8,
    #                                      pin_memory=False, drop_last=True)
    # for it, (data, class_l) in enumerate(data_loader):
    #     print(data, class_l)