# DINGAL
This is a simplified version of implementation for the Dingal paper AAAI 2021.

Key idea:
The key idea of DINGAL paper is that the parameters of previous trained model can be used in the dynamic step, which is similar to the inductive learning in GraphSage. This is a simplified version just for DINGAL-O, which does not have the fine-tuning part and is clean and easy to run. If you are interested in the DINGAL-U for fine-tuning, you just need to add some data preprocessing code (The performances are almost the same for DINGAL-U and DINGAL-O). 

Requirements:
tensorflow 1.3-1.8 cpu

Due to the simplified version, the converge epoch number is set as 2500 for static part. In addition, the reproducing results may have about 1%-3% difference compared with the best performance we have reported in the paper due to simplified version/machine/seeds/training set and testing set splits. 

If you have any problems, feel free to contach me (yucheny5@illinois.edu) and I will share some of my unique findings about this task with you🙃. 

The dataset link: https://drive.google.com/drive/folders/13u-4r4aJbjhUPRbDXrVFA3QfQS0y_8Ye
