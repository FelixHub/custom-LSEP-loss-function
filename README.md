# a custom LSEP loss function with class weights

This is a implementation of the log-sum-exp pairwise (LSEP) loss function for multi-label classification, i took the equation from this great article : https://arxiv.org/abs/1811.05475  (ML-Net: multi-label classification of biomedical texts with deep neural networks).

The LSEP loss function is described as follow :
  
  ![equation](https://raw.githubusercontent.com/FelixHub/custom-LSEP-loss-function/master/equationlsep.png)


where 𝑓(𝑥) is the label prediction function that maps the document vector 𝑥 into K-dimensional label space representing the confidence scores of each label (K equals to number of unique labels). 𝑓u(𝑥i) and 𝑓v(𝑥i) are the 𝑣 and 𝑢 -th element of confidence 
scores for the 𝑖-th instance in the dataset, respectively. 𝑌i is the corresponding label set for the 𝑖-th instance in the dataset.

I had to rewrite the equation to make it more matrix-multiplication compatible, as using if and for loop is a no-no in Keras backend.

![equation](https://raw.githubusercontent.com/FelixHub/custom-LSEP-loss-function/master/equation.png)

I used 3d matrix to do parrallel calculation on each training sample, averaging only at the end.


https://forums.fast.ai/t/lsep-as-a-loss-function/54506
