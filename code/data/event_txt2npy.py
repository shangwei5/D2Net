import os.path
import glob
import numpy as np
from PIL import Image
import cv2
event_path = "./dataset/GOPRO_Random/test/Event"
print('process event data...')
save_result_path = "./dataset/GOPRO_Random/test/Event_npy"
os.makedirs(save_result_path, exist_ok=True)

vid_event_names = sorted(glob.glob(os.path.join(event_path, '*')))


def txt2npy(event_name, h=720, w=1280):
    event_sequence = open(event_name, 'r')  # 打开event
    EVENT = np.loadtxt(event_name)
    start_time = float(EVENT[0][0])
    end_time = float(EVENT[-1][0])
    event_frame = np.zeros([40, h, w], int)
    frame_index = (end_time - start_time) / 20  # 1/20
    time = 1
    for e in event_sequence:
        e = e.rstrip()
        event = e.split()  # t,w,h,p   w和h要颠倒一下
        if float(event[0])-start_time < time * frame_index:
            if time == 21:
                time = 20
            if int(event[3]) > 0:
                event_frame[int(time - 1) * 2 + 1, int(event[2]) - 1, int(event[1]) - 1] = \
                    event_frame[int(time - 1) * 2 + 1, int(event[2]) - 1, int(event[1]) - 1] + 1
            else:
                event_frame[int(time - 1) * 2, int(event[2]) - 1, int(event[1]) - 1] = \
                    event_frame[int(time - 1) * 2, int(event[2]) - 1, int(event[1]) - 1] + 1
        else:
            time = time + 1
            if time == 21:
                time = 20
            if int(event[3]) > 0:
                event_frame[int(time - 1) * 2 + 1, int(event[2]) - 1, int(event[1]) - 1] = \
                    event_frame[int(time - 1) * 2 + 1, int(event[2]) - 1, int(event[1]) - 1] + 1
            else:
                event_frame[int(time - 1) * 2, int(event[2]) - 1, int(event[1]) - 1] = \
                    event_frame[int(time - 1) * 2, int(event[2]) - 1, int(event[1]) - 1] + 1
    event = np.array(event_frame).astype(np.float32)
    event = event / np.max(event)
    # event = event.transpose(1,2,0)

    return event



if __name__ == '__main__':
    for video_index in range(len(vid_event_names)):
        event_dir_names = sorted(glob.glob(os.path.join(vid_event_names[video_index], '*')))
        print(vid_event_names[video_index])
        mat_name = os.path.join(save_result_path, str(vid_event_names[video_index].split('/')[-1]))
        os.makedirs(mat_name, exist_ok=True)
        length = len(event_dir_names)
        print("Length: %d" % length)

        for i in range(length):
            # 提取出对应的event
            event = txt2npy(event_dir_names[i])
            print(event_dir_names[i])
            # print(event.shape)
            np.save(os.path.join(mat_name, event_dir_names[i].split('/')[-1].split('.')[0]+'.npy'), event)
            # save_test_path1 = os.path.join('/home/sw/sw/dataset/Event-data/GOPRO_large_all/test/Random', 'event_all')
            # os.makedirs(save_test_path1, exist_ok=True)
            # for j in range(40):
            #     # print(event[j])
            #     cv2.imwrite(os.path.join(save_test_path1, "%d.png" % j), abs(event[j])*255)
            # event = np.mean(event, 0)#, keepdims=True)
            # print(event.shape)
            # event = np.uint8(abs(event)*255)
            # # event = cv2.equalizeHist(event)
            # cv2.imwrite(os.path.join(save_test_path, "event.png"), event)

        #     if i == 0:
        #         break
        # break




