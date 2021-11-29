# Bringing Events into Video Deblurring with Non consecutively Blurry Frames （ICCV2021）
[[PDF](https://openaccess.thecvf.com/content/ICCV2021/papers/Shang_Bringing_Events_Into_Video_Deblurring_With_Non-Consecutively_Blurry_Frames_ICCV_2021_paper.pdf)]
---
### Introduction
 Existing video deblurring methods assume consecutively blurry frames, while neglecting the fact that sharp frames usually appear nearby blurry frame. In this paper, we develop a principled framework D2Nets for video deblurring to exploit non-consecutively blurry frames, and propose a flexible event fusion module (EFM) to bridge the gap between event-driven and video deblurring.

## Prerequisites
- Python >= 3.6, PyTorch >= 1.1.0, numpy, matplotlib, opencv, imageio, scikit-image, tqdm, cupy
- Requirements: opencv-python, tensorboardX
- Platforms: Ubuntu 20.04, cuda-10.0, 2*2080Ti


## Datasets
  To satisfy our assumption that sharp frames exist in a blurry video, we generate non-consecutively blurry frames in a video by randomly averaging adjacent sharp frames, i.e., the average number is randomly chosen from 1 to 15. And we assume that a generated frame **Bi** is sharp if the number of averaging frames is smaller than 5, i.e., o^i = 1, otherwise o^i = 0. It is worth noting that we randomly generate 50% blurry frames in a video, while the other 50% frames are sharp, without constraining that there must be 2 sharp ones in consecutive 7 frame.

   链接：https://pan.baidu.com/s/1J-vdY1e1jWp6B1AYzcJG6g 
提取码：ho6f 


The code and dataset will be coming soon!





## Cite
If you use any part of our code, or D2Net and non consecutively blurry dataset are useful for your research, please consider citing:
```
  @InProceedings{Shang_2021_ICCV,
      author    = {Shang, Wei and Ren, Dongwei and Zou, Dongqing and Ren, Jimmy S. and Luo, Ping and Zuo, Wangmeng},
      title     = {Bringing Events Into Video Deblurring With Non-Consecutively Blurry Frames},
      booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
      month     = {October},
      year      = {2021},
      pages     = {4531-4540}
  }
```
