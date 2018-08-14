# Tloc
A transfer learning-based random forest regression model developed on Python 2.7

## Packages
numpy <br>
pandas <br>

## Usage
### Data format
|BS_ID|feature_1|feature_2|....|label|
|:---|:---|:---|:---|:---|
|1|-45|4.54|...|121.34|
|2|-52|6.72|...|121.45|

The first column is the domain IDs of the data, and the last column is the ground truth of the data. <br>
You must ensure that source data and target data are in a data file.

### Train a transfer forest regression model
Download Tloc package in your working path. <br>
Import **RF** and **transferForest** modules in your code (See details in run_Example.py). <br>
Specify the target domian ID when initializing. <br>
