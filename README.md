# Commentary-Generator

This is an engine which is a part of a commentary generator. The purpose of this engine is to detect events that take place in basketball, namely 3 pointers, 2 pointers and misses. The engine uses openCV to generate an initial dataset, which is then fed to a convolutional neural network for classification. 

## Note
 - The video dataset is not a part of this repository, but any dataset may be used, with approporiate changes to the code

Execution Instructions
1. Run tracking.py with changes in the code pointing to directory of dataset
2. Run learning.py for a trained neural network. 
