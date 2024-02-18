#  Research Proposal: Lip Reading in English Using Machine Learning and Computer Vision

## I. Title: Lip Reading in English Using Machine Learning and Computer Vision

## II. Introduction:
The purpose of this research project is to harness the power of machine learning and computer vision technologies to advance the field of lip reading. Our primary objective is to develop and deploy robust, real-time systems capable of accurately transcribing spoken language by interpreting the visual cues provided by the movement of the lips. Lip reading has the potential to bridge communication gaps, making spoken language accessible to individuals with hearing impairments, enhancing security through speech recognition in challenging environments, and contributing to a wide range of applications.
## III. Research Question:
How can machine learning and computer vision be used to accurately recognize and transcribe spoken English words from lip movements in real-time?

## IV. Objectives:

Create a comprehensive dataset of video recordings featuring English speakers for training and testing.
Develop a deep learning model capable of recognizing and transcribing spoken English words from lip movements.
Optimize the model's performance, accuracy, and real-time processing capabilities.
Evaluate the system's accuracy through extensive testing, including in real-world scenarios.
## V. Methodology:

The methodology for this research project consists of the following steps:

### A. Data Collection:
this section outlines the methods for collecting the necessary data for our research project.
To train and test our lip reading system, we will employ the following data collection approaches:

1. Recording Video Samples:

   * We will capture video samples of English speakers enunciating a wide range of words and phrases. These recordings will serve as the primary dataset for our research.
   * Each video will be annotated to provide ground truth transcriptions, aligning the spoken words with the corresponding lip movements.

~~2. Utilizing Existing Datasets:~~

   ~~* To augment our dataset, we will consider leveraging publicly available resources. Specifically, we will explore the "Lip Reading Image Dataset" on [Kaggle link](https://www.kaggle.com/datasets/apoorvwatsky/miraclvc1/data), which contains relevant images for lip reading research.~~
   

~~3. Extracting Data from Online Sources:~~

~~4. Visual Speech DataSet:~~

   ~~* We will also consider using the LRS3-TED dataset, which is a large-scale dataset for visual speech recognition. It contains over 100,000 videos of lectures and talks, totaling 1,800 hours of video data. The dataset is available at [LRS3-TED](https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs3.html)(not found).~~
2. Utilizing Existing Datasets:
   * We will also consider using the LRS2 dataset, which is a large-scale dataset for visual speech recognition. It contains over 100,000 videos of lectures and talks, totaling 1,800 hours of video data. The dataset is available at [LRS2](https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs2.html).
   * use the grid corpus dataset, which contains 34 speakers and 1000 sentences. The dataset is available at [grid corpus](https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs2.html).
   
### B. Model Development:

Create a deep learning model, potentially based on convolutional neural networks (CNNs) or recurrent neural networks (RNNs).
Because the dataset is video,we may use LSTM, TCNs ,transformers,or GAANs.
Train the model using the annotated dataset to recognize and transcribe spoken words.

1. Data Preprocessing:

    * We will extract images of the speakers' faces and lips from the video recordings.
    * We will also extract the corresponding audio data from the videos.
    * We will align the audio data with the images to create a dataset of images and corresponding audio files.
2. Dataset Splitting:

    * We will split the dataset into training and testing sets.
    * We will use the training set to train the model and the testing set to evaluate the model's performance.
3. Model Selection(to be decided later, after we have the dataset):

    * We will explore various deep learning models, including convolutional neural networks (CNNs) ; recurrent neural networks (RNNs).such as Long Short-Term Memory(LSTM) networks; Hybrid models that combine CNNs and RNNs; End-to-End Models: models that directly map the input to the output without any intermediate steps; 3D CNNs: models that can capture temporal information from video data.
    * We will select the model that yields the best performance on the testing set.
4. Model Training:
    * I used 3D CNNS conbined with LSTM using the grid corpus dataset. 
5. Model Evaluation:
6. Model Optimization:
7. Combine Audio and Visual Cues:
8. Ethical Considerations:
9. Deployment:
   * deploy the model on jetson nano. 
10. Iterate and Improve:



### C. Performance Optimization:

Fine-tune the model to improve recognition accuracy and reduce processing time.
Implement techniques to handle variations in lighting and background.
### D. Evaluation:

Assess the system's accuracy and real-time performance on a testing dataset.
Conduct real-world testing in different environments and with diverse speakers.
## VI. Timeline:

### semester 1:
Weeks:                  1  |  2  |  3  |  4  |  5  |  6  |  7
-------------------------------------------------------------

Data Collection         [========]

Model Development                [===============]

Performance Optimization            [============]

Evaluation                          [================]

Report Writing                    [=======================]

-------------------------------------------------------------
Weeks:            1  |  2  |  3  |  4  |  5  |  6  |  7
------------------------------------------------------
Data Collection
   - Set up recording equipment [====]
   - Record video samples [====]
   - Annotate collected data [====]
   
Model Development
   - Data preprocessing [=======]
   - Model architecture design [=======]
   - Initial model training [===========]
   - Model fine-tuning [===========]
   
Performance Optimization
   - Implement lighting adjustments [=========]
   - Optimize background handling [=========]
   - Fine-tune hyperparameters [=========]
   
Evaluation
   - Prepare the testing dataset [================]
   - Assess recognition accuracy [================]
   - Real-time performance testing [================]
   
Report Writing
   - Summarize research findings [====================]
   - Document methodology [====================]
   - Present results and analysis [====================]
------------------------------------------------------

**Week 1: Data Collection**

- In the first week, the focus is on data collection.
  - **Set up recording equipment:** You'll set up the necessary recording equipment to capture video samples.
  - **Record video samples:** You'll start recording video samples of English speakers.
  - **Annotate collected data:** Simultaneously, you'll begin annotating the collected data to provide ground truth transcription.

**Week 2: Data Collection (Continued)**

- The second week continues data collection and annotation to ensure a comprehensive dataset.

**Week 3: Model Development**

- In the third week, the model development process begins.
  - **Data preprocessing:** You'll prepare the collected data for training, including cleaning, alignment, and formatting.
  - **Model architecture design:** You'll design the architecture of the deep learning model.
  - **Initial model training:** The initial training of the model using the preprocessed data starts.
  - **Model fine-tuning:** Fine-tuning of the model begins to enhance its performance.

**Week 4: Model Development (Continued)**

- The fourth week is a continuation of model development.

**Week 5: Performance Optimization**

- In the fifth week, the focus shifts to performance optimization.
  - **Implement lighting adjustments:** You'll work on implementing adjustments to handle variations in lighting during video recording.
  - **Optimize background handling:** Techniques to handle variations in the background are implemented.
  - **Fine-tune hyperparameters:** Fine-tuning of the model's hyperparameters for better recognition accuracy.

**Week 6: Evaluation**

- Week 6 is dedicated to evaluating the system.[1]
  - **Prepare the testing dataset:** You'll prepare a separate dataset for testing the model.
  - **Assess recognition accuracy:** The accuracy of the model in recognizing spoken words is assessed.
  - **Real-time performance testing:** The model's real-time performance is evaluated under various conditions.

**Week 7: Report Writing**

- The final week is reserved for report writing.
  - **Summarize research findings:** You'll summarize the key findings from your research.
  - **Document methodology:** The methodology used in data collection, model development, and evaluation is documented.
  - **Present results and analysis:** You'll present the results and analysis in a detailed research report.
### semester 2:
Weeks:                  1   |  2   |  3   |  4  |  5  |  6  |  7 |  8  |
---------------------------------------------------------------------
paper reading           [===========================================]

model developmemt       [=================================]

model training                             [======================]

model evaluation                                      [===========]

report writing                                               [========]

---------------------------------------------------------------------

#### Review of semester 1:
1. go through lots of papers about lip reading and speech recognition.

2. pick pu two datasets for training and testing.

3. set up required environment for training a model.

4. get a hand on experience on grid corpus dataset, by using 3D CNNs conbined with LSTM. the framework is tensorflow.

5. come across two main problems:

    1.1 equipment: My own computer is not compatible for training another dataset which means I can't conduct my own experiments 
by using different Model architecture design.(sloved by buying a new computer.)
    
    1.2 lrw2 dataset is too large to train. It may take several days or longer time to train a model.


#### Plan for semester 2:
1. continue reading papers about lip reading and speech recognition. find out the most suitable model for lip reading.
like vit,GAANs, TCNs, transformers, etc.
2. if I still can't train a model by using lrw2 dataset, I will try to find a pre-trained model and fine-tune it:
if so, I will change my project to "Lip Reading in English Using Machine Learning and Computer Vision Based on Pre-trained Model".
3. find a suitable methodolgy for fine-tuning a pre-trained model like flash attention,Adapter,Adversarial Training, etc.
4. if everything goes well, I will use fine-tuning method to optimize my model.
5. evaluate my model's performance.
6. desin a web application for lip reading.
7. write a report.


**Week 1-8: Paper Reading**
    -[] **continue reading papers about lip reading and speech recognition.**
    - **read papers about 3D CNNs and LSTM.**
    - **read papers about transformers and GAANs.**
    - **read papers about TCNs.**
     - **read papers about vit.**
    - **read papers about pre-trained model.**
    - **read papers about fine-tuning a pre-trained model.**

**Week 3-6: Model Development**
   - **Model architecture design:** design the architecture of the deep learning model through reading papers.
   - **Model training:** The training of the model using the preprocessed data starts.
   - **Model fine-tuning:** Fine-tuning of the model begins to enhance its performance.
   - **Model evaluation:** evaluate the model's performance.

**Week 7-8: Report Writing**
    - **Summarize research findings:**  summarize the key findings from your research.
    - **Document methodology:** The methodology used in data collection, model development, and evaluation is documented.
    - **Present results and analysis:** present the results and analysis in a detailed research report.



## VII. Expected Outcomes:
We anticipate that this research will result in a machine learning-based lip reading system capable of accurately transcribing spoken English words. This technology can significantly benefit the hearing impaired and improve speech recognition in various applications.

## VIII. Significance:

## IX. Conclusion:

## X. References:
[1] M. Abadi, A. Agarwal, P. Barham, E. Brevdo, Z. Chen, C. Citro, G. S. Corrado, A. Davis, J. Dean, M. Devin, et al. Tensorflow: Large-scale machine learning on heterogeneous distributed systems. arXiv preprint arXiv:1603.04467, 2016.
* T. Afouras, J. S. Chung, and A. Zisserman. Deep lip reading: A comparison of models and an online application. In INTERSPEECH, 2018.
* T. Afouras, J. S. Chung, and A. Zisserman. LRS3-TED: a large-scale dataset for visual speech recognition. arXiv preprint arXiv:1809.00496, 2018.
* C. Feichtenhofer, A. Pinz, and A. Zisserman. Convolutional
two-stream network fusion for video action recognition. In
Proceedings of the IEEE Conference on Computer Vision
and Pattern Recognition, 2016.

## XI. Description:

The purpose of this research project is to harness the power of machine learning and computer vision technologies to advance the field of lip reading. Our primary objective is to develop and deploy robust, real-time systems capable of accurately transcribing spoken language by interpreting the visual cues provided by the movement of the lips. Lip reading has the potential to bridge communication gaps, making spoken language accessible to individuals with hearing impairments, enhancing security through speech recognition in challenging environments, and contributing to a wide range of applications.





