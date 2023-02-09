import torch
import torch.nn as nn


class net1(nn.Module):
    def __init__(self):
        super(net1, self).__init__()
        self.conv0 = nn.Conv2d(1, 16, (1, 1), (1, 1), (0, 0))
        self.conv1 = nn.Conv2d(16, 16, (3, 3), (1, 1), (1, 1))
        self.conv2 = nn.Conv2d(32, 32, (5, 5), (1, 1), (2, 2))
        self.conv3 = nn.Conv2d(64, 64, (7, 7), (1, 1), (3, 3))

        self.conv4 = nn.Conv2d(256, 64, (9, 9), (1, 1), (4, 4))
        self.conv5 = nn.Conv2d(64, 64, (7, 7), (1, 1), (3, 3))
        self.conv6 = nn.Conv2d(64, 64, (5, 5), (1, 1), (2, 2))
        self.conv7 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv8 = nn.Conv2d(64, 1, (1, 1), (1, 1), (0, 0))

        self.norm = nn.BatchNorm2d(256)
        self.nom = nn.BatchNorm2d(64)
        self.sig = nn.LeakyReLU()

    def forward(self, x, y):
        temx0 = self.conv0(x)
        temy0 = self.conv0(y)
        temx1 = self.conv1(temx0)
        temy1 = self.conv1(temy0)
        temx2 = self.conv2(torch.cat((temx0, temx1), 1))
        temx3 = self.conv3(torch.cat((temx0, temx1, temx2), 1))
        temy2 = self.conv2(torch.cat((temy0, temy1), 1))
        temy3 = self.conv3(torch.cat((temy0, temy1, temy2), 1))

        tem = torch.cat((temx0, temx1, temx2, temx3, temy0, temy1, temy2, temy3), 1)
        tem = self.norm(tem)
        res1 = self.conv4(tem)
        res1 = self.nom(res1)
        res2 = self.conv5(res1) + res1
        res2 = self.nom(res2)
        res3 = self.conv6(res2) + res1 + res2
        res3 = self.nom(res3)
        res4 = self.conv7(res3) + res1 + res2 + res3
        res4 = self.nom(res4)
        res5 = self.conv8(res4)
        return res5


class net2(nn.Module):
    def __init__(self):
        super(net2, self).__init__()
        self.conv0 = nn.Conv2d(1, 16, (1, 1), (1, 1), (0, 0))
        self.conv1 = nn.Conv2d(16, 16, (3, 3), (1, 1), (1, 1))
        self.conv2 = nn.Conv2d(32, 32, (5, 5), (1, 1), (2, 2))
        self.conv3 = nn.Conv2d(64, 64, (7, 7), (1, 1), (3, 3))

        self.conv4 = nn.Conv2d(128, 64, (9, 9), (1, 1), (4, 4))
        self.conv5 = nn.Conv2d(64, 64, (7, 7), (1, 1), (3, 3))
        self.conv6 = nn.Conv2d(64, 64, (5, 5), (1, 1), (2, 2))
        self.conv7 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv8 = nn.Conv2d(64, 1, (1, 1), (1, 1), (0, 0))

        self.norm = nn.BatchNorm2d(128)
        self.nom = nn.BatchNorm2d(64)
        self.sig = nn.LeakyReLU()

    def forward(self, x):
        temx0 = self.conv0(x)
        temx1 = self.conv1(temx0)
        temx2 = self.conv2(torch.cat((temx0, temx1), 1))
        temx3 = self.conv3(torch.cat((temx0, temx1, temx2), 1))

        tem = torch.cat((temx0, temx1, temx2, temx3), 1)
        tem = self.norm(tem)
        res1 = self.conv4(tem)
        res1 = self.nom(res1)
        res2 = self.conv5(res1) + res1
        res2 = self.nom(res2)
        res3 = self.conv6(res2) + res1 + res2
        res3 = self.nom(res3)
        res4 = self.conv7(res3) + res1 + res2 + res3
        res4 = self.nom(res4)
        res5 = self.conv8(res4)
        return res5


class net3(nn.Module):
    def __init__(self):
        super(net3, self).__init__()
        self.conv0 = nn.Conv2d(1, 16, (1, 1), (1, 1), (0, 0))
        self.conv1 = nn.Conv2d(16, 16, (3, 3), (1, 1), (1, 1))
        self.conv2 = nn.Conv2d(32, 32, (5, 5), (1, 1), (2, 2))
        self.conv3 = nn.Conv2d(64, 64, (7, 7), (1, 1), (3, 3))

        self.conv4 = nn.Conv2d(128, 64, (9, 9), (1, 1), (4, 4))
        self.conv5 = nn.Conv2d(64, 64, (7, 7), (1, 1), (3, 3))
        self.conv6 = nn.Conv2d(64, 64, (5, 5), (1, 1), (2, 2))
        self.conv7 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv8 = nn.Conv2d(64, 1, (1, 1), (1, 1), (0, 0))

        self.norm = nn.BatchNorm2d(128)
        self.nom = nn.BatchNorm2d(64)
        self.sig = nn.LeakyReLU()

    def forward(self, x):
        temx0 = self.conv0(x)
        temx1 = self.conv1(temx0)
        temx2 = self.conv2(torch.cat((temx0, temx1), 1))
        temx3 = self.conv3(torch.cat((temx0, temx1, temx2), 1))

        tem = torch.cat((temx0, temx1, temx2, temx3), 1)
        tem = self.norm(tem)
        res1 = self.conv4(tem)
        res1 = self.nom(res1)
        res2 = self.conv5(res1) + res1
        res2 = self.nom(res2)
        res3 = self.conv6(res2) + res1 + res2
        res3 = self.nom(res3)
        res4 = self.conv7(res3) + res1 + res2 + res3
        res4 = self.nom(res4)
        res5 = self.conv8(res4)
        return res5
