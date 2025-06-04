
# Training loops

&nbsp;&nbsp;&nbsp;&nbsp;This document will serve as logs for the progression of the model between training loops alongside observations, if any.

&nbsp;&nbsp;&nbsp;&nbsp;Previously, there were other model versions with smaller datasets before the current ones, but due to inexperience, I had to redo the process from the beginning a couple of times since the datasets were defective.

## Initial training (v1)

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

## Training Loop 1 (v2)

Pre-training from initial, this training loop includes some corrections for unknown data to the first version.

Additional messages: 323

```
F1 Score: 0.9334260385885624
Precision: 0.9391395592864638
Recall: 0.927781617138908
Accuracy: 0.8739693757361602
ROC AUC: 0.9429842964264138
```

## Training Loop 2 (v3)

Pre-training from the last version, newly added data consists in the continous data collection over from discord augmented through back to back translation and word swapping.

```
F1 Score: 0.9264778936910084
Precision: 0.9304839514385498
Recall: 0.9225061830173125
Accuracy: 0.8682447521214828
ROC AUC: 0.9415311367035816
```

### Model evaluation comparison on the current dataset

Comparing v1, v2 and v3
By all parameters, v3 has better scores.

```
Model v1
F1 Score: 0.8879028882377564
Precision: 0.9018707482993197
Recall: 0.874361088211047    
Accuracy: 0.7994640464493077 
ROC AUC: 0.8988466234225745

Model v2
F1 Score: 0.9122428666224287
Precision: 0.9178768152228343
Recall: 0.9066776586974443   
Accuracy: 0.8456900401965163 
ROC AUC: 0.9303969731984164  

Model v3
F1 Score: 0.9264778936910084
Precision: 0.9304839514385498
Recall: 0.9225061830173125   
Accuracy: 0.8682447521214828 
ROC AUC: 0.9415311367035816
```

### Comparing v3 with another version of BERT fine-tuned
Using [bert-base-romanian-cased-v1](https://huggingface.co/dumitrescustefan/bert-base-romanian-cased-v1) to fine-tune a model on opjustice-lm dataset for comparison.

Do note that v3 is fine-tuned on RoBERT-small, so while my model will be a lot more lightweight and easier to train, will perform worse in metrics.

```
F1 Score: 0.9312116691529919
Precision: 0.9361773037827029
Recall: 0.9262984336356141
Accuracy: 0.8729343456900402
ROC AUC: 0.9441678976126585
```

## Training loop 3 (v4)
1300 rows were augmented and added in two more ways, that being swapping words and back to back translation.
test was also augmented, therefore a decrease in scoring is due to generating artificial data never encountered in the training loop before, that also lacks similarity with other data due to the fact that the model is trained on Romanian language mainly, so the behavior is unpredictible when it encounters untranslated words due to augmentation.

```
F1 Score: 0.9160874238624149
Precision: 0.9144492131616595
Recall: 0.9177315147164393
Accuracy: 0.8602069614299154
ROC AUC: 0.937151290352151
```

### Comparing v4-small with v4-large
The main model has Ro-BERT-small as base, for version 4, we are comparing a fine-tuned Ro-BERT-large with the small model.

```
LARGE
F1 Score: 0.9275341480948958
Precision: 0.928869690424766
Recall: 0.9262024407753051
Accuracy: 0.8805268109125117
ROC AUC: 0.9439848701891032
```

Time required for predicting: 2m36s(large) and 50s(small)
Memory size: 3.83GB(large) and 222MB(small)

It can be observed that although the larger model takes 3.12 times longer and uses 17.25 times more memory than the smaller one, its scoring is only better by a few hundredths.

### Comparing v4 attention mask vs v4 unmasked
Batch size: 32

Epoches: 8
```
MASKED
F1 Score: 0.920045851841238
Precision: 0.9182039182039182
Recall: 0.9218951902368988
Accuracy: 0.8652869238005645
ROC AUC: 0.9390270642749735

UNMASKED
F1 Score: 0.9187562688064193
Precision: 0.9169169169169169
Recall: 0.9206030150753769
Accuracy: 0.8641580432737536
ROC AUC: 0.9383795007749696
```

### Comparing v4 with different classification NLPs on its dataset

```
RoBERT-small
F1 Score: 0.3091198303287381
Precision: 0.24506094997898276
Recall: 0.41852117731514715   
Accuracy: 0.09595484477892756 
ROC AUC: 0.47900436717009703
```

```
RoBERT-large
F1 Score: 0.3253486540332669
Precision: 0.24018333561362704
Recall: 0.504091888011486
Accuracy: 0.011665098777046096
ROC AUC: 0.49925547231041073
```

### Comparing v4 Sequence Classifier (CLS) with CNN fine-tuned model
```
Model v4
F1 Score: 0.920045851841238
Precision: 0.9182039182039182
Recall: 0.9218951902368988
Accuracy: 0.8652869238005645
ROC AUC: 0.9390270642749735
```

```
Model CNN (v4)
F1 Score: 0.8838681824722981
Precision: 0.8859079763450166
Recall: 0.88183776022972
Accuracy: 0.7952963311382879
ROC AUC: 0.9048207626939002
```