from utils import pil_loader, Distortions
import torch
import torchvision

image = pil_loader('/data5/zyl/patch_camelyon/valid/tumor/406.png')
distor = Distortions()
distortions = [
    'pixelate', 'jpeg_compression', 'defocus_blur',
    'motion_blur', 'brightness', 'saturate',
    'hue', 'marking_blur', 'bubble_blur']

img_collection = []
for distortion in distortions:
    distortion = getattr(distor, distortion)
    for severity in range(1,6):
        distored_image = distortion(image, severity)
        distored_image = torchvision.transforms.functional.to_tensor(distored_image)
        img_collection.append(distored_image)

img_collection = torch.stack(img_collection)
torchvision.utils.save_image(torchvision.utils.make_grid(img_collection, 5),"./vis_sample.png")





