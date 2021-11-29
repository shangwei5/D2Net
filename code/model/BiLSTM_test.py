import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

def print_network(net):
	num_params = 0
	for param in net.parameters():
		num_params += param.numel()
	print(net)
	print('Total number of parameters: %d' % num_params)

#Set iteration time
ITERATION = 4

#Model
class BiLSTM_resnet152(nn.Module):
    def __init__(self, opt):
        super(BiLSTM_resnet152, self).__init__()
        self.iteration = opt.seq_num
        self.use_GPU = opt.use_gpu
        self.device = 'cuda' if opt.use_gpu else 'cpu'

        self.conv_e = nn.Sequential(
            nn.Conv2d(20, 3, 3, 1, 1),
            #nn.BatchNorm2d(32),
            nn.LeakyReLU()
            )
        self.extractor = torchvision.models.resnet152(pretrained=True)
        # self.avg_pool = nn.AdaptiveAvgPool2d(1)
        numFit = self.extractor.fc.in_features
        self.extractor.fc = nn.Sequential(nn.Linear(numFit, numFit // 4))
        self.bilstm = nn.LSTM(numFit // 4, numFit // 4, 1, bidirectional=True, bias=False)
        self.fc = nn.Sequential(
            nn.Linear(numFit//2, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, imgs):
        score_list = []
        hidden_vec = None
        img_vec = []

        for index in range(self.iteration):

            x = self.extractor.conv1(imgs[index])
            x = self.extractor.bn1(x)
            x = self.extractor.relu(x)
            x = self.extractor.maxpool(x)

            x = self.extractor.layer1(x)
            x = self.extractor.layer2(x)
            x = self.extractor.layer3(x)
            x = self.extractor.layer4(x)
            x = self.extractor.avgpool(x)
            x = torch.flatten(x, 1)
            img_feat = F.normalize(x)
            img_vec.append(img_feat)

            feat = self.extractor.fc(x)
            # feat = self.extractor(imgs[index])
            batch, channel = feat.size()
            feat = feat.view(1, batch, channel)  # 1,b,c
            if hidden_vec is None:
                hidden_vec = feat
            else:
                hidden_vec = torch.cat((hidden_vec, feat), 0)

        # print(hidden_vec.shape)  # seq_num, b, c
        hidden_out, _ = self.bilstm(hidden_vec)
        # print(hidden_out.shape)  # seq_num, b, c*2

        for i in range(hidden_out.shape[0]):
            feature = hidden_out[i]
            # print(feature.shape)   #b, c*2
            score = self.fc(feature)
            score_list.append(score)

        return score_list, img_vec


