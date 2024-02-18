# week plan for Feb 18th - Feb 23th

## works have done
- [x] went through lots of papers about lip reading 
- [x] got the model I have trained on the grid corpus dataset
- [x] got intuitive understanding of the technical terms in deep learning
- [x] set up the environment for the model
- [x] got some training logs of the model like the loss and accuracy
- [x] added resNet50 and resNet18 to try on the basic architecture of the model  achieved better performance above 98%
accuracy (need to be approved)


## Research Plan 
- [ ] implement the model I have trained on the grid corpus dataset
  - [ ] use openCV to detect the mouth area 
  - [ ] padding the mouth area with time dimension to fit the model

## problems when implementing the model I need to solve
- [ ] what is the best practices to pad the mouth area with time dimension: 
  - [ ] I trained the model by specifying time length of the mouth area(75 frames), but the real time
    length of the mouth area is not fixed. So Do I need to pad the mouth area with the same frame for several times?
    or if the mouth area is too short, do I need to pad the mouth area with the same frame using zero padding?
  - [ ] Is that affect the performance of the model because the real speech is not cut off at the time length I pad 
  leading the speech is not complete?
- [ ] How to make the train input data flexible to fit the model:
  -[ ] Does the training model accept a flexible input data?
  -[ ] If not, how to make the input data flexible to fit the model?
- [x] When training the model what are the logs I need to pay attention to? I only have the logs, val_loss and accuracy
try TensorFlow Visualization Toolkit (TensorBoard) and tensorboardx to visualize the training logs

