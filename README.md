# a custom LSEP loss function with class weights

This is a implementation of the log-sum-exp pairwise (LSEP) loss function for multi-label classification, i took the equation from this great article : https://arxiv.org/abs/1811.05475  (ML-Net: multi-label classification of biomedical texts with deep neural networks).

The LSEP loss function is described as follow :



where 𝑓(𝑥) is the label prediction function that maps the document vector 𝑥 into Kdimensional label space representing the confidence scores of each label (K equals to number of unique labels). 𝑓3(𝑥5) and 𝑓6(𝑥5) are the 𝑣 and 𝑢 -th element of confidence 
scores for the 𝑖-th instance in the dataset, respectively. 𝑌 5 is the corresponding label set for the 𝑖-th instance in the dataset.
