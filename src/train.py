import os
import torch
import datetime
import argparse
from dataloader import vimeo_septuplet
from torch.utils.data import DataLoader
import models
from tester import UVG_8Frames
from trainer import Trainer

parser = argparse.ArgumentParser(description='DVC-pytorch')

# Model Selection
parser.add_argument('--seed', type=int, default=17)
parser.add_argument('--use_cuda', type=bool, default=True)
parser.add_argument('--gpu_idx', type=str, default='0')
parser.add_argument('--model', type=str, default='meanscale')
parser.add_argument('--intra', type=str, default='H265L23')

# Directory Setting
parser.add_argument('--train', type=str, default='/home/chajin/dataset/vimeo_septuplet')
parser.add_argument('--out_dir', type=str, default='./experiment/test-meanscale')
parser.add_argument('--load', type=str, default='./experiment/Meanscale-4096-H265-20/checkpoint/model_epoch060.pth')
parser.add_argument('--test_input', type=str, default='/home/chajin/dataset/UVG_8Frames_444')
parser.add_argument('--test_every', type=int, default=5)

# Learning Options
parser.add_argument('--epochs', type=int, default=50, help='Max Epochs')
parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
parser.add_argument('--lmbda', type=float, default=512, help='loss function configuration')
parser.add_argument('--patch_size', type=int, default=256, help='Patch size')
parser.add_argument('--test_height_size', type=int, default=1024, help='Test image height size')
parser.add_argument('--test_width_size', type=int, default=1920, help='Test image width size')

# Optimization specifications
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--aux_lr', type=float, default=0.001, help='auxiliary learning rate')
parser.add_argument('--lr_decay', type=int, default=40, help='learning rate decay per N epochs')
parser.add_argument('--decay_type', type=str, default='step', help='learning rate decay type')
parser.add_argument('--gamma', type=float, default=0.1, help='learning rate decay factor for step decay')
parser.add_argument('--optimizer', default='ADAM', choices=('SGD', 'ADAM', 'RMSprop', 'ADAMax'), help='optimizer to use (SGD | ADAM | RMSprop | ADAMax)')
parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
parser.add_argument('--clip_max_norm', type=float, default=1, help='maximum gradient magnitude')


def main():
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_idx

    dbTrain = vimeo_septuplet(args.train, random_crop=(args.patch_size, args.patch_size))
    dbTest = UVG_8Frames(args)
    loaderTrain = DataLoader(dataset=dbTrain, batch_size=args.batch_size, shuffle=True, num_workers=0)
    model = models.Model(args)

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    intStartEpoch = 0
    if args.load is not None:
        checkpoint = torch.load(args.load)
        model.load(checkpoint['state_dict'])
        intStartEpoch = checkpoint['epoch']

    my_trainer = Trainer(args, loaderTrain, dbTest, model, intStartEpoch)
    my_trainer.load_state_dict()

    now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    with open(args.out_dir + '/config.txt', 'a') as f:
        f.write(now + '\n\n')
        for arg in vars(args):
            f.write('{}: {}\n'.format(arg, getattr(args, arg)))
        f.write('\n')

    while not my_trainer.terminate():
        my_trainer.train()
        my_trainer.test()

    my_trainer.close()


if __name__ == "__main__":
    main()
