# Dataset preparation

Download the LSP dataset and extract it into './datasets/LSP'

# Training on LSP Dataset
## Training fully supervised baseline
python train_hpe.py

Check config parameters within the main() function. You can specify the number of labelled samples used for training. The path to the dataset can be specified as well. Model weights and training information will be logged in './runs'

## Training a single pseudo-label iteration
python train_ssl.py

Check config parameters within the main() function. Besides the specification of the number of labelled sampled and the path to the dataset, you also have to specify the path to the weights to the teacher model, which will be used to calculate pseudo-labels.
