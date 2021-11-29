import torch
import torch.nn as nn
from model import recons_video
from model import flow_pwc
from utils import utils


def make_model(args):
    device = 'cpu' if args.cpu else 'cuda'
    load_flow_net = True
    load_recons_net = False
    flow_pretrain_fn = args.pretrain_models_dir + 'network-default.pytorch'
    recons_pretrain_fn = ''
    is_mask_filter = True
    return D2Net(in_channels=args.n_colors, n_sequence=args.n_sequence, out_channels=args.n_colors,
                    n_resblock=args.n_resblock, n_feat=args.n_feat,
                    load_flow_net=load_flow_net, load_recons_net=load_recons_net,
                    flow_pretrain_fn=flow_pretrain_fn, recons_pretrain_fn=recons_pretrain_fn,
                    is_mask_filter=is_mask_filter, device=device)


class D2Net(nn.Module):

    def __init__(self, in_channels=3, n_sequence=3, out_channels=3, n_resblock=3, n_feat=32,
                 load_flow_net=False, load_recons_net=False, flow_pretrain_fn='', recons_pretrain_fn='',
                 is_mask_filter=False, device='cuda'):
        super(D2Net, self).__init__()
        print("Creating D2Net Net")

        self.n_sequence = n_sequence
        self.device = device

        # assert n_sequence == 5, "Only support args.n_sequence=5; but get args.n_sequence={}".format(n_sequence)

        self.is_mask_filter = is_mask_filter
        print('Is meanfilter image when process mask:', 'True' if is_mask_filter else 'False')
        extra_channels = 0
        print('Select mask mode: concat, num_mask={}'.format(extra_channels))

        self.flow_net_near = flow_pwc.Flow_PWC(load_pretrain=load_flow_net, pretrain_fn=flow_pretrain_fn, device=device)
        self.flow_net_nsf = flow_pwc.Flow_PWC(load_pretrain=load_flow_net, pretrain_fn=flow_pretrain_fn, device=device)
        self.recons_net_near = recons_video.RECONS_VIDEO(in_channels=in_channels, n_sequence=3, out_channels=out_channels,
                                                    n_resblock=n_resblock, n_feat=n_feat,
                                                    extra_channels=extra_channels)
        self.recons_net_nsf = recons_video.RECONS_VIDEO(in_channels=in_channels, n_sequence=3,
                                                         out_channels=out_channels,
                                                         n_resblock=n_resblock, n_feat=n_feat,
                                                         extra_channels=extra_channels)
        if load_recons_net:
            self.recons_net_near.load_state_dict(torch.load(recons_pretrain_fn))
            self.recons_net_nsf.load_state_dict(torch.load(recons_pretrain_fn))
            print('Loading reconstruction pretrain model from {}'.format(recons_pretrain_fn))


    def forward(self, x, isTraining=False):
        frame_list = [x[:, i, :, :, :] for i in range(self.n_sequence)]
        left_pre_sharp_frame = x[:, self.n_sequence, :, :, :]
        left_sub_sharp_frame = x[:, self.n_sequence+1, :, :, :]
        right_pre_sharp_frame = x[:, self.n_sequence + 2, :, :, :]
        right_sub_sharp_frame = x[:, self.n_sequence + 3, :, :, :]

        # Interation 1
        warped_l0, _, _, flow_mask_l0 = self.flow_net_nsf(frame_list[0], left_pre_sharp_frame)
        warped_r0, _, _, flow_mask_r0 = self.flow_net_nsf(frame_list[0], left_sub_sharp_frame)
        warped_lm, _, _, flow_mask_lm = self.flow_net_nsf(frame_list[self.n_sequence//2], left_sub_sharp_frame)
        warped_rm, _, _, flow_mask_rm = self.flow_net_nsf(frame_list[self.n_sequence//2], right_pre_sharp_frame)
        warped_l2, _, _, flow_mask_l2 = self.flow_net_nsf(frame_list[2], right_pre_sharp_frame)
        warped_r2, _, _, flow_mask_r2 = self.flow_net_nsf(frame_list[2], right_sub_sharp_frame)

        states_m = None
        concated = torch.cat([warped_l0, frame_list[0], warped_r0], dim=1)
        recons_1, states_m, _, _ = self.recons_net_nsf(concated, states_m)

        concated = torch.cat([warped_lm, frame_list[self.n_sequence//2], warped_rm], dim=1)
        recons_2, states_m, _, _ = self.recons_net_nsf(concated, states_m)

        concated = torch.cat([warped_l2, frame_list[2], warped_r2], dim=1)
        recons_3, states_m, _, _ = self.recons_net_nsf(concated, states_m)

        # Interation 2
        warped12, _, _, flow_mask12 = self.flow_net_near(recons_2, recons_1)
        warped32, _, _, flow_mask32 = self.flow_net_near(recons_2, recons_3)

        concated = torch.cat([warped12, recons_2, warped32], dim=1)
        out, _, _, _ = self.recons_net_near(concated)

        if isTraining:
            return recons_1, recons_2, recons_3, out
        else:
            return out
