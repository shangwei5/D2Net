import os
import glob
import re
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from model import BiLSTM_test
import imageio

def main():
    parser = argparse.ArgumentParser(description="Detector_train")
    parser.add_argument("--save_path", type=str, default="./logs", help='path to save models and log files')
    parser.add_argument("--save_label_path", type=str, default="../dataset/GOPRO_Random/test/detector_label", help='path to label')
    parser.add_argument("--test_path", type=str, default="../dataset/GOPRO_Random/test", help='path to testing data')
    parser.add_argument("--use_gpu", type=bool, default=True, help='use GPU or not')
    parser.add_argument("--gpu_id", type=str, default="0, 1", help='GPU id')
    parser.add_argument("--input_type", type=str, default="img", help='Type of input data')
    parser.add_argument('--seq_num', type=int, default=5, help='sequence num')
    parser.add_argument('--size_must_mode', type=int, default=4, help='number of color channels to use')
    parser.add_argument('--n_threads', type=int, default=4, help='number of threads for data loading')
    parser.add_argument('--n_colors', type=int, default=3, help='number of color channels to use')
    parser.add_argument('--rgb_range', type=int, default=1, help='maximum value of RGB')
    opt = parser.parse_args()
    opt.n_sequence = 5
    opt.n_frames_per_video = 100
    if opt.use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
    model = BiLSTM_test.BiLSTM_resnet152(opt)
    model = nn.DataParallel(model)
    os.makedirs(opt.save_label_path, exist_ok=True)
    # Move to GPU
    if opt.use_gpu:
        model = model.cuda()

    # load the lastest model
    model.load_state_dict(torch.load(os.path.join(opt.save_path, 'net_epoch11.pth')))

    print("Loading testing dataset path: %s\n" % opt.test_path)
    model.eval()
    n_frames_video = []
    dir_input = os.path.join(opt.test_path, 'blur')
    dir_label = os.path.join(opt.test_path, 'label')
    vid_input_names = sorted(glob.glob(os.path.join(dir_input, '*')))
    vid_label_names = sorted(glob.glob(os.path.join(dir_label, '*')))
    images_input = []
    images_label = []
    for vid_input_name, vid_label_name in zip(vid_input_names, vid_label_names):
        input_dir_names = sorted(glob.glob(os.path.join(vid_input_name, '*')))
        label_dir_names = np.load(vid_label_name)
        images_input.append(input_dir_names)
        images_label.append(label_dir_names)

        n_frames_video.append(len(input_dir_names))

    num_video = len(images_input)
    num_frame = sum(n_frames_video) - (opt.n_sequence - 1) * len(n_frames_video)
    print("Number of videos to load:", num_video)
    print("Number of frames to load:", num_frame)

    data_input = []
    n_videos = len(images_input)
    for idx in range(n_videos):
        print("Loading video %d" % idx)
        inputs = np.array([imageio.imread(lr_name) for lr_name in images_input[idx]])
        data_input.append(inputs)
        labels = images_label[idx]
        label = torch.from_numpy(labels).float()
        with torch.no_grad():
            n_poss_frames = [n - opt.n_sequence + 1 for n in n_frames_video]
            output = []
            for frame_idx in range(n_poss_frames[idx]):
                inputs = data_input[idx][frame_idx:frame_idx + opt.n_sequence]
                # labels = images_label[idx][frame_idx:frame_idx + opt.n_sequence]
                inputs_list = [inputs[i, :, :, :] for i in range(opt.n_sequence)]
                inputs_concat = np.concatenate(inputs_list, axis=2)
                inputs_concat = get_patch(inputs_concat, opt.size_must_mode)
                inputs_list = [inputs_concat[:, :, i * opt.n_colors:(i + 1) * opt.n_colors] for i in
                               range(opt.n_sequence)]
                inputs = np.array(inputs_list)
                input_tensors = np2Tensor(*inputs, rgb_range=opt.rgb_range, n_colors=opt.n_colors)
                # label = torch.from_numpy(labels).float()[np.newaxis, :]
                inputs = torch.stack(input_tensors)[np.newaxis, :, :, :, :]
                # print(inputs.size())
                img1 = Variable(inputs[:, 0, :, :, :])
                img2 = Variable(inputs[:, 1, :, :, :])
                img3 = Variable(inputs[:, 2, :, :, :])
                img4 = Variable(inputs[:, 3, :, :, :])
                img5 = Variable(inputs[:, 4, :, :, :])

                if opt.use_gpu:
                    img1, img2, img3, img4, img5 = img1.cuda(), img2.cuda(), img3.cuda(), img4.cuda(), img5.cuda()
                    label = label.cuda()

                img_list = []
                img_list.append(img1)
                img_list.append(img2)
                img_list.append(img3)
                img_list.append(img4)
                img_list.append(img5)

                socres, img_vec = model(img_list)

                detect_result = []
                for n in range(len(socres)):
                    if socres[n][:, 0].data > 0.5:
                        score = 0
                    elif socres[n][:, 1].data > 0.5:
                        score = 1
                    detect_result.append(score)

                if frame_idx == 0:
                    output.append(detect_result[0])
                    output.append(detect_result[1])
                    output.append(detect_result[2])
                elif frame_idx == n_poss_frames[idx]-1:
                    output.append(detect_result[2])
                    output.append(detect_result[3])
                    output.append(detect_result[4])
                else:
                    output.append(detect_result[2])
            result = sum(1 for m, n in zip(output, label) if (m == n))
            acc = result/len(label)
            name = os.path.split(os.path.dirname(images_input[idx][0]))[-1]
            print(name + " Avg_acc: %.6f" % acc)
            np.save(os.path.join(opt.save_label_path, name + ".npy"), output)

def get_patch(input, size_must_mode=1):
    h, w, c = input.shape
    new_h, new_w = h - h % size_must_mode, w - w % size_must_mode
    input = input[:new_h, :new_w, :]
    return input

def np2Tensor(*args, rgb_range=255, n_colors=1):
    def _np2Tensor(img):
        img = img.astype('float64')
        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))  # HWC -> CHW
        tensor = torch.from_numpy(np_transpose).float()  # numpy -> tensor
        tensor.mul_(rgb_range / 255)  # (0,255) -> (0,1)

        return tensor

    return [_np2Tensor(a) for a in args]

if __name__ == "__main__":

    main()
