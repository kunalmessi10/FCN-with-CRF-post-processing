# FCN-in-Pytorch
Pytorch based implementation of [Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/abs/1411.4038) CVPR'15 paper.

## Requirements
1. pytorch >=0.3
2. pydensecrf 
3. opencv>3.0
4. matplotlib
5. numpy
6. scikit-image
7. scikit-learn

Download the dataset from this link and place it in the main directory: https://github.com/mostafaizz/camvid .
Dataset contains the following files:
1.  701_StillsRaw_full
2. LabelApproved_full
3.  label_colors.txt  
4.  label_colorsSorted.txt

Preprocessing

```python
python camvid_utils.py
``` 
It will create a folder in the directory which contains the above folders,a directory named Labeled_idx which would contain the 32 label encoded numpy vectors.

Training

```python
python camvid_train.py
``` 
Training Results
Coming Soon!!


