import os
import glob
import numpy as np
import imageio
import torch
import torch.utils.data as data
import utils.utils as utils


class VIDEODATA(data.Dataset):
    def __init__(self, args, name='', train=True):
        self.args = args
        self.name = name
        self.train = train
        self.n_seq = args.n_sequence
        self.n_frames_per_video = args.n_frames_per_video
        print("n_seq:", args.n_sequence)
        print("n_frames_per_video:", args.n_frames_per_video)

        self.n_frames_video = []

        if train:
            self._set_filesystem(args.dir_data)
        else:
            self._set_filesystem(args.dir_data_test)

        self.images_gt, self.images_input, self.images_label, self.PreSIndex_label, self.SubSIndex_label = self._scan()   #

        self.num_video = len(self.images_gt)
        self.num_frame = sum(self.n_frames_video) - (self.n_seq - 1) * len(self.n_frames_video)
        print("Number of videos to load:", self.num_video)
        print("Number of frames to load:", self.num_frame)

        if train:
            self.repeat = max(args.test_every // max((self.num_frame // self.args.batch_size), 1), 1)
            print("Dataset repeat:", self.repeat)

        if args.process:
            self.data_gt, self.data_input, self.data_label = self._load(self.images_gt, self.images_input, self.images_label)   #

    def _set_filesystem(self, dir_data):
        print("Loading {} => {} DataSet".format("train" if self.train else "test", self.name))
        self.apath = dir_data
        self.dir_gt = os.path.join(self.apath, 'gt')
        self.dir_input = os.path.join(self.apath, 'blur')
        self.dir_label = os.path.join(self.apath, 'label')
        print("DataSet GT path:", self.dir_gt)
        print("DataSet INPUT path:", self.dir_input)
        print("DataSet label path:", self.dir_label)

    def return_BlurryIndices(self, detect_result):
        """find all Blurry frames and their pre/sub sharp frames."""
        SharpIndex_list = [i for i in range(len(detect_result)) if detect_result[i] == 1]
        # print("Length of blur frames: %d" % len([i for i in range(len(detect_result)) if detect_result[i] == 0]))
        pre_index = 0
        sub_index = 1
        PreSIndex_list, SubSIndex_list = [], []
        for i in range(len(detect_result)):
            if i < SharpIndex_list[pre_index]:
                PreSIndex_list.append(SharpIndex_list[pre_index])
                SubSIndex_list.append(SharpIndex_list[pre_index])
            elif i == SharpIndex_list[pre_index]:
                PreSIndex_list.append(i)
                SubSIndex_list.append(i)  # SharpIndex_list[sub_index]
            elif i > SharpIndex_list[pre_index] and i < SharpIndex_list[sub_index]:
                PreSIndex_list.append(SharpIndex_list[pre_index])
                SubSIndex_list.append(SharpIndex_list[sub_index])
            elif i == SharpIndex_list[sub_index]:
                pre_index = pre_index + 1
                sub_index = sub_index + 1
                if sub_index > len(SharpIndex_list) - 1:
                    sub_index = sub_index - 1
                    pre_index = pre_index - 1
                PreSIndex_list.append(i)  # SharpIndex_list[pre_index]
                SubSIndex_list.append(i)  # SharpIndex_list[sub_index]
            elif i > SharpIndex_list[sub_index]:
                PreSIndex_list.append(SharpIndex_list[sub_index])
                SubSIndex_list.append(SharpIndex_list[sub_index])

        return PreSIndex_list, SubSIndex_list

    def _scan(self):
        vid_gt_names = sorted(glob.glob(os.path.join(self.dir_gt, '*')))
        vid_input_names = sorted(glob.glob(os.path.join(self.dir_input, '*')))
        vid_label_names = sorted(glob.glob(os.path.join(self.dir_label, '*')))
        assert len(vid_gt_names) == len(vid_input_names) == len(vid_label_names), "len(vid_gt_names) must equal len(vid_input_names)"  #

        images_gt = []
        images_input = []
        images_label = []
        PreSIndex_label, SubSIndex_label = [], []

        for vid_gt_name, vid_input_name, vid_label_name in zip(vid_gt_names, vid_input_names, vid_label_names):  #
            if self.train:
                gt_dir_names = sorted(glob.glob(os.path.join(vid_gt_name, '*')))[:self.args.n_frames_per_video]
                input_dir_names = sorted(glob.glob(os.path.join(vid_input_name, '*')))[:self.args.n_frames_per_video]
                label_dir_names = np.load(vid_label_name)[:self.args.n_frames_per_video]
            else:
                gt_dir_names = sorted(glob.glob(os.path.join(vid_gt_name, '*')))
                input_dir_names = sorted(glob.glob(os.path.join(vid_input_name, '*')))
                label_dir_names = np.load(vid_label_name)
            PreSIndex_list, SubSIndex_list = self.return_BlurryIndices(label_dir_names.squeeze().tolist())
            images_gt.append(gt_dir_names)
            images_input.append(input_dir_names)
            images_label.append(label_dir_names)
            PreSIndex_label.append(PreSIndex_list)
            SubSIndex_label.append(SubSIndex_list)
            self.n_frames_video.append(len(gt_dir_names))

        return images_gt, images_input, images_label, PreSIndex_label, SubSIndex_label

    def _load(self, images_gt, images_input, images_label):  #
        data_input = []
        data_gt = []
        # video_num, single_video_frame_num, h,w,c

        n_videos = len(images_gt)
        for idx in range(n_videos):
            if idx % 10 == 0:
                print("Loading video %d" % idx)
            gts = np.array([imageio.imread(hr_name) for hr_name in images_gt[idx]])
            inputs = np.array([imageio.imread(lr_name) for lr_name in images_input[idx]])
            data_input.append(inputs)
            data_gt.append(gts)

        return data_gt, data_input, images_label

    def __getitem__(self, idx):
        if self.args.process:
            inputs, gts, labels, filenames = self._load_file_from_loaded_data(idx)   #
        else:
            inputs, gts, labels, filenames = self._load_file(idx)   #

        inputs_list = [inputs[i, :, :, :] for i in range(self.n_seq+4)] #2
        inputs_concat = np.concatenate(inputs_list, axis=2)
        gts_list = [gts[i, :, :, :] for i in range(self.n_seq)]
        gts_concat = np.concatenate(gts_list, axis=2)
        inputs_concat, gts_concat = self.get_patch(inputs_concat, gts_concat, self.args.size_must_mode)  #
        inputs_list = [inputs_concat[:, :, i*self.args.n_colors:(i+1)*self.args.n_colors] for i in range(self.n_seq+4)]  #2
        gts_list = [gts_concat[:, :, i*self.args.n_colors:(i+1)*self.args.n_colors] for i in range(self.n_seq)]

        inputs = np.array(inputs_list)
        gts = np.array(gts_list)

        input_tensors = utils.np2Tensor(*inputs, rgb_range=self.args.rgb_range, n_colors=self.args.n_colors)
        gt_tensors = utils.np2Tensor(*gts, rgb_range=self.args.rgb_range, n_colors=self.args.n_colors)
        label_tensors = torch.from_numpy(labels).float()

        return torch.stack(input_tensors), torch.stack(gt_tensors), label_tensors, filenames  #

    def __len__(self):
        if self.train:
            return self.num_frame * 2 #self.repeat
        else:
            return 60 #self.num_frame-2

    def _get_index(self, idx):
        if self.train:
            return idx % self.num_frame
        else:
            return idx

    def _find_video_num(self, idx, n_frame):
        for i, j in enumerate(n_frame):
            if idx < j:
                return i, idx
            else:
                idx -= j

    def _load_file(self, idx):
        idx = self._get_index(idx)

        n_poss_frames = [n - self.n_seq + 1 for n in self.n_frames_video]
        video_idx, frame_idx = self._find_video_num(idx, n_poss_frames)
        f_labels = self.images_label[video_idx][frame_idx:frame_idx + self.n_seq]
        f_gts = self.images_gt[video_idx][frame_idx:frame_idx + self.n_seq]
        f_inputs = self.images_input[video_idx][frame_idx:frame_idx + self.n_seq]
        f_inputs.append(self.images_input[video_idx][self.PreSIndex_label[video_idx][frame_idx]])
        f_inputs.append(self.images_input[video_idx][self.SubSIndex_label[video_idx][frame_idx]])
        f_inputs.append(self.images_input[video_idx][self.PreSIndex_label[video_idx][frame_idx+self.n_seq-1]])
        f_inputs.append(self.images_input[video_idx][self.SubSIndex_label[video_idx][frame_idx+self.n_seq-1]])
        gts = np.array([imageio.imread(hr_name) for hr_name in f_gts])
        inputs = np.array([imageio.imread(lr_name) for lr_name in f_inputs])
        filenames = [os.path.split(os.path.dirname(name))[-1] + '.' + os.path.splitext(os.path.basename(name))[0]
                     for name in f_inputs]

        return inputs, gts, f_labels, filenames

    def _load_file_from_loaded_data(self, idx):
        idx = self._get_index(idx)

        n_poss_frames = [n - self.n_seq + 1 for n in self.n_frames_video]
        video_idx, frame_idx = self._find_video_num(idx, n_poss_frames)
        gts = self.data_gt[video_idx][frame_idx:frame_idx + self.n_seq]
        inputs = self.data_input[video_idx][frame_idx:frame_idx + self.n_seq]
        labels = self.data_label[video_idx][frame_idx:frame_idx + self.n_seq]
        filenames = [os.path.split(os.path.dirname(name))[-1] + '.' + os.path.splitext(os.path.basename(name))[0]
                     for name in self.images_gt[video_idx][frame_idx:frame_idx + self.n_seq]]

        return inputs, gts, labels, filenames

    def get_patch(self, input, gt, size_must_mode=1):   #
        if self.train:
            input, gt = utils.get_patch(input, gt, patch_size=self.args.patch_size)   #
            h, w, c = input.shape
            new_h, new_w = h - h % size_must_mode, w - w % size_must_mode
            input, gt = input[:new_h, :new_w, :], gt[:new_h, :new_w, :]
            if not self.args.no_augment:
                input, gt = utils.data_augment(input, gt)   #
        else:
            h, w, c = input.shape
            new_h, new_w = h - h % size_must_mode, w - w % size_must_mode
            input, gt = input[:new_h, :new_w, :], gt[:new_h, :new_w, :]
        return input, gt
