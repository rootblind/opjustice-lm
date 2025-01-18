
# Training loops

&nbsp;&nbsp;&nbsp;&nbsp;This document will serve as logs for the progression of the model between training loops alongside observations, if any.

&nbsp;&nbsp;&nbsp;&nbsp;Previously, there were other model versions with smaller datasets before the current ones, but due to inexperience, I had to redo the process from the beginning a couple of times since the datasets were defective.

## Initial training

Below is the score comparison between version 2 of a model trained on a more defective dataset vs. the version pre-trained from RoBERT-small on a more accurate and higher-quality dataset.

```
Average: micro

Model version1
F1 Score: 0.9148974451241454
Precision: 0.9411438089950028
Recall: 0.8900752669350603   
Accuracy: 0.8310972688068998 
ROC AUC: 0.9098615359010884  


Model old
F1 Score: 0.8873226544622426
Precision: 0.9299693016116654
Recall: 0.8484158935760546   
Accuracy: 0.7690464781983709
ROC AUC: 0.8539663428343907
```

