# README
This repository contains the official implementation of the methods presented in the paper [**Exploring Vulnerabilities of No-Reference Quality Assessment Models: A Query-Based Black-Box Method**](https://ieeexplore.ieee.org/document/10614617). This paper provides a query-base black-box attacking method for NR-IQA models. 
This repository contains the attack code for both a single image and a set of images.

Our gratitude extends to [Surfree](https://github.com/t-maho/SurFree) and [MBS](https://github.com/jimmie33/MBS) for providing the code of classification attack framework and sliency detection, respectively.

## Dependencies
- python
- torch
- torchvision
- Pillow
- numPy
- scipy
- opencv-python
- h5py
- glob
- argparse


## Usage
Download the files for attacked images from the LIVE dataset and other related information from [Google Drive](https://drive.google.com/drive/folders/1x1LMwYx9E_ZjBq1aC7t1wWC3hm3NLnMh?usp=share_link) and put them into the dataset and checkpoints folders respectively.

To attack a single image, use the following command:
```
python attack_demo.py --incr
```
The option "incr" provides a strategy which increases the predicted score of the attacked image.


To attack the whole dataset, use the command in attack_live_dbcnn.sh, or use the following command:
```
sh attack_live_dbcnn.sh
```

To test the attack performance on the image set, use the following command:
```
python test_performance.py
```

## Citation
```
@ARTICLE{blackboxIQA,
  author={Yang, Chenxi and Liu, Yujia and Li, Dingquan and Jiang, Tingting},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  title={Exploring Vulnerabilities of No-Reference Image Quality Assessment Models: A Query-Based Black-Box Method}, 
  year={2024},
  doi={10.1109/TCSVT.2024.3435865}}
```