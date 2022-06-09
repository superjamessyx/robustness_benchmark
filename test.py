import os
import argparse
import torch
import torchvision.datasets as dset
import torchvision.transforms as trn
import networks
import numpy as np
import sys
sys.path.append("./ImageNet-C/create_c")
from utils import DistortImageFolder, print_log, sort_metric

parser = argparse.ArgumentParser()
parser.add_argument('-model_name', default='shufflenet_v2_x0_5', help="The name of the backbone")
parser.add_argument('-num_classes', default=2, type=int, help="The number of classes")
parser.add_argument('-ckpt', default='best', type=str, help="checkpoint name")
parser.add_argument('-save_path', default='/data3/zyl/robustness-master/wandb/run-20220209_070650-34bupus4', help="Path to saved checkpoint.")
parser.add_argument('-device', default=5, type=int, help="CUDA device")
parser.add_argument('-prefetch', default=8, type=int, help="Number of workers(processings)")
parser.add_argument('-test_bs', default=256, type=int, help="Test batch size")
args = parser.parse_args()

# logger
logger = open(os.path.join(args.save_path, 'test_logger.txt'), 'a')
print_log('save path : {}'.format(args.save_path), logger)

# Device
device = 'cuda:{}'.format(args.device)

# model
print_log("Backbone: {}".format(args.model_name), logger)
net = networks.SimpleNet(args.model_name, num_classes=args.num_classes).to(device)
ckpt = torch.load(os.path.join(args.save_path, 'saved_models/%s-model.pth'%args.ckpt))
net.load_state_dict(ckpt)
net.eval()

# transformer
test_transform = trn.Compose([
    trn.ToTensor(),
    trn.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

""" test """
# /////////////// Data Loader ///////////////
test_dataset = dset.ImageFolder(root='/data5/zyl/patch_camelyon/test/', transform=test_transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_bs,
    shuffle=False, num_workers=args.prefetch, pin_memory=True, drop_last=False)

correct = 0
collected_preds = torch.zeros((len(test_dataset), 2)).to(device)
collected_targs = torch.zeros(len(test_dataset)).to(device)
for batch_idx, (data, target) in enumerate(test_loader):
    with torch.no_grad():
        data, target = data.to(device), target.to(device)
        collected_targs[batch_idx * args.test_bs:batch_idx * args.test_bs + data.shape[0]] = target
        output = net(data)
        output = torch.softmax(output, dim=1)
        collected_preds[batch_idx * args.test_bs:batch_idx * args.test_bs + data.shape[0], :] = output
        conf, pred = torch.max(output, dim=1)

        correct += pred.eq(target).sum()

clean_error = 1 - correct / len(test_dataset)
print_log('Test dataset error (%): {:.2f}'.format(100 * clean_error), logger)

""" clean validation """
# /////////////// Data Loader ///////////////
clean_dataset = dset.ImageFolder(root='/data5/zyl/patch_camelyon/valid/', transform=test_transform)
clean_loader = torch.utils.data.DataLoader(clean_dataset, batch_size=args.test_bs,
    shuffle=False, num_workers=args.prefetch, pin_memory=True, drop_last=False)

conf_collection = torch.zeros((len(clean_dataset), 6)).to(device)

correct = 0
for batch_idx, (data, target) in enumerate(clean_loader):
    with torch.no_grad():
        data, target = data.to(device), target.to(device)
        output = net(data)
        output = torch.softmax(output, dim=1)

        conf, pred = torch.max(output, dim=1)
        conf_collection[batch_idx * args.test_bs:batch_idx * args.test_bs + data.shape[0], 0] = conf

        correct += pred.eq(target).sum()

clean_error = 1 - correct / len(clean_loader.dataset)
print_log('Clean dataset error (%): {:.2f}'.format(100 * clean_error), logger)

""" noise validation """
def show_performance(distortion_name):
    errs = []
    for severity in range(1, 6):
        distorted_dataset = DistortImageFolder(root='/data5/zyl/patch_camelyon/valid/', method=distortion_name, severity=severity,
            transform=test_transform)

        distorted_dataset_loader = torch.utils.data.DataLoader(
            distorted_dataset, batch_size=args.test_bs, shuffle=False, num_workers=args.prefetch, pin_memory=True)

        correct = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(distorted_dataset_loader):
                data, target = data.to(dtype=torch.float, device=device), target.to(device)
                output = net(data)
                output = torch.softmax(output, dim=1)
                conf, pred = torch.max(output, dim=1)
                conf_collection[batch_idx * args.test_bs:batch_idx * args.test_bs + data.shape[0], severity] = conf
                correct += pred.eq(target).sum()
            correct = correct.item()

            errs.append(1 - 1.*correct / len(distorted_dataset))

    print_log('\n=Error Average %s'%(tuple(errs),), logger)

    rank_err = sort_metric(conf_collection.cpu().detach().numpy())
    print_log('=Rank Error Average %s' %rank_err, logger)
    return np.mean(errs), rank_err

distortions = [
    'pixelate', 'jpeg_compression', 'defocus_blur',
    'motion_blur', 'brightness', 'saturate',
    'hue', 'marking_blur', 'bubble_blur']

error_rates = []
rank_errors = []
for distortion_name in distortions:
    error, rank_error = show_performance(distortion_name)
    error_rates.append(error)
    rank_errors.append(rank_error)
    print_log('=Distortion: {:15s}  | CE (unnormalized) (%): {:.2f}'.format(distortion_name, 100 * error), logger)

print_log('\nmCE (unnormalized by AlexNet errors) (%): {:.2f}'.format(100 * np.mean(error_rates)), logger)
print_log('ECE (unnormalized by AlexNet errors) (%): {:.2f}'.format(100 * np.mean(rank_errors)), logger)


