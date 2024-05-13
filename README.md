# Background

The lip-reading project is my master's research project at the Technological University of the Shannon. This project
primarily focuses on employing neural networks within the realm of computer vision to predict speech based on lip
movements. The goal is to deepen my understanding of deep learning and computer vision. While I did not contribute to
the design of the system's architecture, I have gained hands-on experience by reading relevant papers and implementing
the system.

if you want to know more about the project, please refer to [lip reading](Lip%20Reading.md)

# Introduction

Essentially, my approach involves studying academic papers to grasp the underlying concepts and strategies of the
technology. I then attempt to implement the model on my own. Whenever I encounter challenges, I refer back to the
original source code and revisit the papers to deepen my understanding of the concepts, model, and code.

## File directory Structure

```bash
├── Deep_Learning_learning
├── images
├── Literature review 
├── README.md
├── requirements.txt
└── src
    ├── lipNet           #  this is from https://github.com/nicknochnack/LipNet.git
    │   ├── app
    │   ├──│   ├── modelutil.py  # load model
    │   ├──│   ├── streamlitapp.py # design the web app
    │   ├──│   ├── utils.py  # process the video and text
    │   ├── data 
    │   ├──│   ├── alignments
    │   ├──│   ├── s1
    │   ├── models   # the path to saved model
    ├── pytorch_lipNet   # this is I developed
    │   ├── config.py       # the configuration file
    │   ├── dataset.py     # load the processed data
    │   ├── data_loader.py # return the data loader 
    │   ├── main.py        # train the model
    │   ├── model       # the model architecture
    │   ├── pretrain.py   # extract the mouth frames from the video
    │   ├── utils         # the utility functions
    │   └── video_to_frames.py # extract the mouth frames from the video
    └── lipNet_word_level   # haven't finished yet
    └── lipNet_tensorflow   # haven't finished yet
```

from the above file directory, you can see that the project is divided into three parts: learning, trying and results.
Literature review adn Deep_Learning_learning are the learning part, src is the trying part, and images are the results
part.

# Installation

If you have the interest to run the code, you can follow the steps below to install the required packages.

## lipNet from  https://github.com/nicknochnack/LipNet.git

### dependencies

```bash
python3 -m venv venv
pip install -r requirements.txt
```

**Note**:

* when installing the pytorch or tensorflow, you may come across some errors, recommend you to install it from
  [pytoch](https://pytorch.org/get-started/locally/) and [tensorflow](https://www.tensorflow.org/install/pip)
* many errors may occur the Nvida driver, you can refer to
  the [Nvida driver](https://www.nvidia.com/Download/index.aspx)

### dataset

download the dataset [GRID](https://zenodo.org/records/3625687) I only use the s1 dataset.

### run the code

```bash
cd src/lipNet
```

pick up a *.ipynb file to run you will see the detailed steps.

## pytorch_lipNet

### dependencies

Same as above

### dataset

Same as above

### run the code

make sure you data directory is in the config.py file
and other parameters are set up correctly

```bash
python pretrain.py
```

This will extract the mouth frames from the video and save it into npy file

```bash
python main.py
```

This will train the model and save the model into the models directory

# Things I have not done

- [ ] Although I have successfully simulated LipNet using TensorFlow, I encountered difficulties when attempting to
  implement the same model in PyTorch, despite the theoretical compatibility of the approach.
- [ ] Thus, my attempts to predict words at the word level also failed.
- [ ] Due to the failure with the PyTorch version, I have not modified the original model design. My plan is to enhance
  the model by incorporating an attention mechanism and integrating a language model to improve prediction accuracy
- [ ] My ultimate objective is to develop a fusion model that combines audio and visual information to enhance
  prediction accuracy. This will involve transitioning from predicting character spelling to predicting pronunciation,
  making the system's output more human-like.











