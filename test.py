from __future__ import print_function
import argparse
import torch
from torch.autograd import Variable
from PIL import Image
from torchvision.transforms import ToTensor
from glob import glob
import numpy as np
import time
import imageio
parser = argparse.ArgumentParser(description='Jun-fuse')
parser.add_argument('--model1', default='./model/model1_epoch_200.pth')
parser.add_argument('--model2', default='./model/model2_epoch_200.pth')
parser.add_argument('--model3', default='./model/model3_epoch_200.pth')
opt = parser.parse_args()
print(opt)


def process(out, cb, cr):
    out_img_y = out.data[0].numpy()
    out_img_y *= 255.0
    out_img_y = out_img_y.clip(0, 255)
    out_img_y = Image.fromarray(np.uint8(out_img_y[0]), mode='L')
    out_img = Image.merge('YCbCr', [out_img_y, cb, cr]).convert('RGB')
    return out_img


def main():
    images_list1 = glob('images/mrt1-pet' + '/*mri.jpg')
    images_list2 = glob('images/mrt1-pet' + '/*pet.jpg')
    name1 = []
    name2 = []
    model1 = torch.load(opt.model1).cuda()
    model2 = torch.load(opt.model2).cuda()
    model3 = torch.load(opt.model3).cuda()
    index = 0
    for i, image_path in enumerate(images_list1):
        name1.append(image_path)
    for i, image_path in enumerate(images_list2):
        name2.append(image_path)

    for i in enumerate(images_list1):
        img1 = Image.open(name1[index]).convert('L')
        img0 = Image.open(name2[index]).convert('YCbCr')

        y1 = img1
        y0, cb0, cr0 = img0.split()
        LR1 = y1
        LR0 = y0
        LR1 = Variable(ToTensor()(LR1)).view(1, -1, LR1.size[1], LR1.size[0])
        LR0 = Variable(ToTensor()(LR0)).view(1, -1, LR0.size[1], LR0.size[0])
        LR1 = LR1.cuda()
        LR0 = LR0.cuda()
        with torch.no_grad():
            tem = model1(LR0, LR1)
            tem = model2(tem)
            tem = model3(tem)
            tem = process(tem.cpu(), cb0, cr0)
            imageio.imsave('results/' + name2[index], tem)
            index += 1


if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print(end - start)
