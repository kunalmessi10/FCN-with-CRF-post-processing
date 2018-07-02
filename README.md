# FCN-in-Pytorch
Pytorch based implemenatation of Fully Convolutional Networks

Firstly for the dataset go to this link: https://github.com/mostafaizz/camvid
Clone this and it would come in the format of a folder name camvid-master.It would contain the following folders:
1  701_StillsRaw_full
2  LabelApproved_full
and two text files namely label_colors.txt and label_colorsSorted.txt

The required dependencies are:
1. pytorch >=0.3
2. pydensecrf 
3. opencv>3.0
4. matplotlib
5. numpy
6. scikit-image
7. scikit-learn

Now first run the file camvid_utils first.
It will create a folder in the directory which contains the above folders,a directory named Labeled_idx which would contain the 32 label encoded numpy vectors.

After this run the camvid_train.py and the model will start training.


