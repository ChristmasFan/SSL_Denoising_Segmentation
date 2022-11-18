## Dataset preparation

Download the PASCAL VOC dataset and extract it into './datasets/pascal'
Download the augmented PASCAL VOC dataset and extract it into './datasets/pascal_aug'

## Training on PASCAL VOC Dataset
# Training fully supervised baseline
python train_seg.py

Check config parameters within the main() function. You can specify the number of labelled samples used for training. The path to the dataset can be specified as well. Model weights and training information will be logged in './runs'

# Training a single pseudo-label iteration
Python train_pl_iteration.py
Check config parameters within the main() function. Besides the specification of the number of labelled sampled and the path to the dataset, you also have to specify the path to the weights to the teacher model, which will be used to calculate pseudo-labels.
