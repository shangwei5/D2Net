# Bringing Events into Video Deblurring with Non consecutively Blurry Frames （ICCV2021）
[[PDF](https://openaccess.thecvf.com/content/ICCV2021/papers/Shang_Bringing_Events_Into_Video_Deblurring_With_Non-Consecutively_Blurry_Frames_ICCV_2021_paper.pdf)]
---
### Introduction
 Existing video deblurring methods assume consecutively blurry frames, while neglecting the fact that sharp frames usually appear nearby blurry frame. In this paper, we develop a principled framework D2Nets for video deblurring to exploit non-consecutively blurry frames, and propose a flexible event fusion module (EFM) to bridge the gap between event-driven and video deblurring.

## Prerequisites
- Python >= 3.6, PyTorch >= 1.1.0
- Requirements: opencv-python, numpy, matplotlib, imageio, scikit-image, tqdm, cupy
- Platforms: Ubuntu 20.04, cuda-10.0, 2*2080Ti

## Updatas
[2022-01-14] Our D2Net model trained on consecutively blurry GOPRO dataset can be download [Here](https://pan.baidu.com/s/1SKnyk7EPuWqeQkUBWKKZUg)[password: blrf]


## Datasets
  GOPRO_Random: To satisfy our assumption that sharp frames exist in a blurry video, we generate non-consecutively blurry frames in a video by randomly averaging adjacent sharp frames, i.e., the average number is randomly chosen from 1 to 15. And we assume that a generated frame **Bi** is sharp if the number of averaging frames is smaller than 5, i.e., label is 1, otherwise label is 0. It is worth noting that we randomly generate 50% blurry frames in a video, while the other 50% frames are sharp, without constraining that there must be 2 sharp ones in consecutive 7 frame.

### Dataset Organization Form
```
|--dataset
    |--blur  
        |--video 1
            |--frame 1
            |--frame 2
                ：  
        |--video 2
            :
        |--video n
    |--gt
        |--video 1
            |--frame 1
            |--frame 2
                ：  
        |--video 2
         :
        |--video n
    |--Event
        |--video 1
            |--frame 1
            |--frame 2
                ：  
        |--video 2
         :
        |--video n
    |--label
        |--video 1
        |--video 2
         :
        |--video n
```

## Download
Please download the testing datasets from [BaiduYun](https://pan.baidu.com/s/1J-vdY1e1jWp6B1AYzcJG6g)[password: ho6f] and the training datasets from [BaiduYun](https://pan.baidu.com/s/1RemasH9D9qYubruGhuZF8w)[password: ju3q]. And pretrained PWCFlow model can be downloaded [Here](https://pan.baidu.com/s/1EIPwGE_M9dsMic-ena6JVg)[password: wkt0] and Our D2Net model trained on non-consecutively blurry GOPRO dataset can be download [Here](https://pan.baidu.com/s/151M8IZCIgGv1iwbb4Bi0HA)[password: 16fr]

_(i)  If you have downloaded the pretrained models，please put PWC_Flow model to './pretrain_models' and  D2Net model to './code/logs', respectively._

_(ii) If you have downloaded the datasets，please put them to './dataset'._

## Getting Started

### 1) Testing
```
bash Inference.sh
```
The results on GOPRO_Random are also available at [BaiduYun](https://pan.baidu.com/s/1v81oA9Tvq_zwMUMeceLVLQ)[password: d3as]

*Note:
The results in our paper is testing on 4X down sampling GOPRO due to the large event data. Now we have retrained our method on original resolution of GOPRO.*

###
Metrics(PSNR/SSIM) calculating codes are [Here](https://github.com/csbhr/OpenUtility#chapter-calculating-metrics).
Average PSNR/SSIM values on GOPRO_Random dataset:

Dataset        |   DMPHN   |   STFAN   |  CDVD-TSP |   ESTRNN  |   D2Net   |   D2Net*
---------------|-----------|-----------|-----------|-----------|-----------|-----------
GOPRO_Random   |32.09/0.897|31.76/0.873|33.76/0.925|33.52/0.912|35.53/0.948|35.72/0.951


For testing consecutively blurry dataset, please put D2Net official model to './code/offical_logs'
```
bash Inference_official.sh
```


### 2) Training
Without event data:
```
python main_d2net.py --template D2NET
```
With event data:
```
python main_d2net_event.py --template D2NET_EVENT
```
Please check the path for you dataset.




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

## Acknowledgements

This code is built on [CDVD-TSP](https://github.com/csbhr/CDVD-TSP). We thank the authors for sharing the codes.
