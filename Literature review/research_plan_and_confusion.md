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
- [x] what is the best practices to pad the mouth area with time dimension: (pading with zero)
  - [x] I trained the model by specifying time length of the mouth area(75 frames), but the real time
    length of the mouth area is not fixed. So Do I need to pad the mouth area with the same frame for several times?
    or if the mouth area is too short, do I need to pad the mouth area with the same frame using zero padding?(cut off 
  the video by use opencv or ffmpeg)
  - [x] Is that affect the performance of the model because the real speech is not cut off at the time length I pad 
  leading the speech is not complete?(yes)
- [x] How to make the train input data flexible to fit the model: (not yet)
  -[x] Does the training model accept a flexible input data?
  -[x] If not, how to make the input data flexible to fit the model?
- [x] When training the model what are the logs I need to pay attention to? I only have the logs, val_loss and accuracy
try TensorFlow Visualization Toolkit [TensorBoard](https://www.tensorflow.org/tensorboard/get_started) and 
[TensorBoardX](https://github.com/lanpa/tensorboardX) to visualize the training logs   

# week














# Tip of the project
## training process
1. when you are designing the training process, you will generate a lot of datas like preprocess data which you can use it 
next time, and you don't want to generate it every train time. So you can save the preprocess data like 'pickle' in python
```python
import pickle
data=[[1,2,3,4,5],[6,7,8,9,10]]
with open('data.pickle', 'wb') as f:
    pickle.dump(data, f)
```
Example:
when you aught to tranform the mouth area to a series of frames, you can save the frames to a pickle file, and next time
you can load the pickle file directly, which can save a lot of time.
```python
import pickle
with open('data.pickle', 'rb') as f:
    data = pickle.load(f)
```

2. when you are training the model, you may want to slightly change the architecture of the model,
making all the changes in a file, like config.py or config.json,and import the file in the main file, 
so you can easily change the architecture of the model without changing the main file.


