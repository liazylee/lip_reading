## Abstract:

In this investigation, we examined the impact of image resolution on the performance of lip-reading models, with a focus
on character error rate (CER) and word error rate (WER) as primary metrics. Two models were rigorously trained, one at a
low resolution of 35x70 pixels and the other at a higher resolution of 70x140 pixels.The dataset we are using is one of
the grid corpus datasets, which is a widely used dataset in the field of lip reading. The results were telling: the
high-resolution model achieved a CER of 0.0008 and a WER of 0.0033, in contrast to the low-resolution model's CER of
0.3034 and WER of 0.3015. This indicates a significant performance leap with higher resolution inputs. However, the
trade-off for precision was computational time, with the higher resolution training taking 4.319 hours, compared to
1.212 hours for the lower resolution. To translate these findings into practical insights, we have developed an online
platform where users can gauge the model's accuracy in real-time scenarios. The direct comparison of CER and WER across
resolutions highlights the delicate balance between model accuracy and computational efficiency in the field of
automated lip-reading.

## Introduction:

Lip reading is a complex visual task that involves deciphering speech from the movements of the lips, tongue, and face
without auditory input. It is particularly critical for those with hearing impairments and has extensive applications in
noisy environments where audio signals may be compromised. The advent of deep learning has significantly enhanced the
prospects of automating this task, but achieving high accuracy depends on multiple factors including the resolution of
the input data.

In this paper, we investigate the influence of input resolution on the performance of a lip-reading model that
integrates three-dimensional convolutional neural networks (Conv3D), Long Short-Term Memory networks (LSTM), and
Connectionist Temporal Classification (CTC) for training. This combination leverages spatial feature extraction through
Conv3D, temporal dynamics captured by LSTM, and an end-to-end training approach offered by CTC, which is well-suited for
sequence prediction problems such as lip reading.

Our study compares the character error rate (CER) and word error rate (WER) of two models: one trained on
low-resolution (35x70 pixels) and the other on high-resolution (70x140 pixels) datasets. These metrics gauge the models'
abilities to accurately recognize and reproduce characters and words from visual information alone. By presenting a
side-by-side analysis of models trained with varying input resolutions, we aim to illuminate the trade-offs between
computational efficiency and the precision of lip-reading models. Furthermore, the paper discusses the implementation of
an online platform to assess the practical capabilities of the trained models in real-world scenarios.

## Experimental Setup:

## Results:

## Conclusion:

We trained two models on the grid corpus dataset, one at a low resolution of 35x70 pixels and the other at a higher
resolution of 70x140 pixels. each model was trained for 100 epochs, with a batch size of 4 and a learning rate of
0.0001.
The high-resolution model achieved a significantly lower CER of 0.0008 and WER of 0.0033, compared to the low-resolution
model's CER of 0.3034 and WER of 0.3015. as shown in the table below, the high-resolution model outperformed the
low-resolution
model across all metrics, indicating the importance of input resolution in lip-reading tasks.

### Dataset Analysis:

We review datasets that have received substantial attention, delineating their characteristics and exploring
dataset-related challenges. Additionally, we examine solutions presented in the retrospective literature and
survey metrics used for evaluating VSR systems.

### Feature Extraction:

We analyze the feature extraction process, focusing on the input preparation and feature extraction modules.
We discuss the challenges associated with these modules and the techniques used to address them.

### VSR Pipeline Scrutiny:

For each sub-module of the VSR pipeline, we analyze obstacles hindering progress and system accuracy.
We explore how current methods address and alleviate these challenges.

### Future Directions:

A detailed overview of open problems and potential future directions in the field of lip reading is presented.

## Lip Reading Workflow:

generally, lip reading is the process of inferring speech from visual information. In other words, lip reading learns
from video, i.e., series of images/frames, and predicts the corresponding speech. The process of lip reading can
be divided into two main steps: feature extraction and classification.
The feature extraction step is responsible for extracting the most relevant information from the video frames.
The classification step is responsible for predicting the corresponding speech from the extracted features.
The following figures shows the general lip reading pipeline.
![img.png](../images/process_of_lip_reading.png)

![img_1.png](../images/flows_process.png)

All lip reading always come with data collection first,which is the very first step in "Input Video" in the above
figure.
And then, we need to do some preprocessing to extract the features like face detection ,face alignment, and mouth
extraction.
After that, it comes to the most important step, matching the features with the corresponding speech. which is called
Region-of-Interest
(ROI).The ROI is the part of the image that contains the mouth. The ROI is then fed to the feature extraction module,
which extracts
the most relevant information from the ROI. The extracted features are then fed to the classification module, which
predicts the
corresponding speech. Simply, this step is aligning a sequence of images with a sequence of words.

when we get aligned images and words, we can train a model to predict the corresponding words from the images. During
this part, luckily,
we get lots of models to choose from, like CNN, RNN, LSTM, 3D CNN, residual networks, HPC,2D DenseNet,
3DMM,GANs,BANNs,Reinfoecement learning,
P3D,MIM,SpotFast,Deformation Flow, and so on. The richness of model options underscores the flexibility in choosing an
architecture tailored
to specific requirements, a topic that will be expounded upon later in this discussion. Once aligned images and
corresponding words are obtained, model training ensues
. A plethora of models allows for a nuanced approach, accommodating various complexities inherent in lip
reading.Following training, we can test the model with the test set.
The test set is the part of the dataset that the model has not seen during training. The test set is used to evaluate
the performance of the model. The performance of the model is
measured using metrics like accuracy, word error rate, and sentence error rate. The accuracy is the percentage of
correctly predicted words.
The word error rate is the percentage of words that are incorrectly predicted. The sentence error rate is the percentage
of sentences that
are incorrectly predicted. The higher the accuracy, the better the model. The lower the word error rate, the better the
model. The lower
the sentence error rate, the better the model.

### Data Collection:

The initial step is the collection of a substantial dataset comprising videos with corresponding transcriptions
(Assael et al., 2016; Nagrani et al., 2020). It is imperative for this dataset to be diverse, encompassing
variations in speakers, lighting conditions, and other pertinent variables. This diversity ensures the model's
adaptability to real-world scenarios and enhances its generalization capabilities.

### Preprocessing:

Following data collection, preprocessing is employed to isolate the speaker's mouth region, recognizing it as the
primary focal point for lip reading (Chung & Zisserman, 2016; Assael et al., 2016). This step involves the extraction
of relevant visual information, facilitating a focused analysis on the crucial articulatory movements during speech.

### Feature Extraction:

The heart of the lip reading process lies in feature extraction. Artificial Intelligence (AI) algorithms are employed
to analyze a sequence of image frames derived from silent talking videos (Wand et al., 2020). Visual and temporal
features are extracted, capturing the nuances of lip and facial movements that convey speech-related information.
This step is pivotal for providing the subsequent model with rich input for accurate analysis.

### Model Training:

With the extracted features in hand, a deep learning model is trained to decipher the sequential patterns
inherent in lip movements (Petridis et al., 2018; Zhou et al., 2019). This model may comprise a
Convolutional Neural Network (CNN) for image recognition, coupled with a Recurrent Neural Network (RNN)
like Long Short-Term Memory (LSTM) for sequence prediction. The training process involves mapping the
sequence of features to meaningful speech units, such as characters, words, or phrases. The model
learns to recognize and interpret the visual and temporal cues inherent in the lip movements,
enabling it to make predictions based on the learned patterns.

### Evaluation and Testing:

Once the model is trained, it undergoes evaluation on a distinct test set of videos that it has not encountered
during training (Chung & Zisserman, 2016; Assael et al., 2016). This rigorous evaluation process assesses the
model's ability to generalize its lip reading capabilities to new and unseen data. The predictions made by the
model are compared with the actual words spoken in the videos to measure its accuracy. This step is essential
for validating the model's efficacy and ensuring its reliability in practical applications.

The culmination of these processes represents a comprehensive approach to lip reading, leveraging deep learning
techniques to bridge the gap between visual information extracted from silent talking videos and meaningful
speech understanding. The dynamic interplay between data collection, preprocessing, feature extraction, model
training, and evaluation converges to form a sophisticated system capable of decoding spoken language solely
through visual cues. As advancements in technology and datasets continue, the potential for even more accurate
and versatile lip reading systems becomes increasingly promising.

## Datasets and Performance Evaluation:

## Automatic Lip Reading:

## Future Directions:

## Conclusions:

## References:

* Assael, Y. M., Shillingford, B., Whiteson, S., & de Freitas, N. (2016). LipNet: End-to-End Sentence-level Lipreading.
  arXiv preprint arXiv:1611.01599.
* Chung, J. S., & Zisserman, A. (2017). Lip Reading in the Wild. In Asian Conference on Computer Vision (pp. 87-104).
  Springer.
* Chung, J. S., & Zisserman, A. (2016). Lip Reading Sentences in the Wild. In Asian Conference on Computer Vision (pp.
  87-103). Springer.
* McGurk, H., & MacDonald, J. (1976). Hearing lips and seeing voices. Nature, 264(5588), 746-748.
* Petridis, S., Stavropoulos, G., Bastiaan Kleijn, W., & Cirstea, C. (2018). End-to-end audiovisual speech recognition.
  IEEE Transactions on Neural Networks and Learning Systems, 29(12), 6261-6270.
* Summerfield, Q. (1992). Lipreading and audio-visual speech perception. Philosophical Transactions of the Royal Society
  of London. Series B: Biological Sciences, 335(1273), 71-78.
* Sumby, W. H., & Pollack, I. (1954). Visual contribution to speech intelligibility in noise. The Journal of the
  Acoustical Society of America, 26(2), 212-215.
* Cooke, M., Barker, J., Cunningham, S., & Shao, X. (2006). An audio-visual corpus for speech perception and automatic
  speech recognition. The Journal of the Acoustical Society of America, 120(5), 2421-2424.
* Nagrani, A., Chung, J. S., & Zisserman, A. (2020). VoxCeleb: A large-scale speaker identification dataset. In ICASSP
  2020-2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) (pp. 3084-3088). IEEE.
  [1] J. K. Chorowski, D. Bahdanau, D. Serdyuk, K. Cho, and Y. Bengio. Attention-based models for speech recognition. In
  Advances in Neural Information Processing Systems, pages 577–585, 2015.
  [2] J. S. Chung et al., "Lip Reading in the Wild," ACM Transactions on Graphics, vol. 36, no. 4, Article 31, July
  2017.
  [3] C. C. Cook et al., "GRID Corpus: A Multimodal Dataset for Research in Automatic Lip-Reading," Proc. of the
  International Conference on Auditory-Visual Speech Processing, 2006.
  [4] J. S. Chung and A. Zisserman, "Lip Reading Sentences in the Wild," Computer Vision and Pattern Recognition, 2016.
  [5] A. Afouras et al., "Deep Audio-Visual Speech Recognition," IEEE Transactions on Pattern Analysis and Machine
  Intelligence, 2018.
  [6] N. Harte and E. Gillen, "TCD-TIMIT: An Audio-Visual Corpus of Continuous Speech," IEEE Transactions on Multimedia,
  vol. 17, no. 5, May 2015.
  [7] L. Matthews et al., "Extracting Visual Features for Lipreading," IEEE Transactions on Pattern Analysis and Machine
  Intelligence, vol. 24, no. 2, Feb 2002.
  [8] Y. Ephrat et al., "Looking to Listen at the Cocktail Party: A Speaker-Independent Audio-Visual Model for Speech
  Separation," ACM Transactions on Graphics, vol. 37, no. 4, Article 112, August 2018.
  [9]A. Ben-Hamadou et al., "MIRACL-VC1: A Multi-Speaker Visual Corpus for Lip-Based Speaker Verification," IEEE
  International Conference on Acoustics, Speech and Signal Processing, 2014.
  [10] P. Ma, S. Petridis, M. Pantic, "End-to-End Audio-Visual Speech Recognition with Conformers," Department of
  Computing, Imperial College London, 2021.
  [11] Y. Assael, B. Shillingford, S. Whiteson, N. de Freitas, "LipNet: End-to-End Sentence-level Lipreading," 2016.
  [12] B. Shillingford et al., "Large-Scale Visual Speech Recognition," DeepMind Technologies, 2018.
  [13] E. Battenberg et al., "Exploring Neural Transducers for End-to-End Speech Recognition," Google Research, 2017.
  [14] K. A. Lee et al., "Large-Scale Visual Speech Recognition," DeepMind and Google, 2018.
  [15] T. Afouras, J. S. Chung, A. Senior, O. Vinyals, A. Zisserman, "Deep Audio-Visual Speech Recognition," 2018.
  [16] J. S. Chung, A. Senior, O. Vinyals, A. Zisserman, "Lip Reading Sentences in the Wild," Computer Vision and
  Pattern
  Recognition (CVPR), 2017.
  [17] Y. Ephrat et al., "Looking to Listen at the Cocktail Party: A Speaker-Independent Audio-Visual Model for Speech
  Separation," ACM Transactions on Graphics, vol. 37, no. 4, Article 112, August 2018.
  [18] A. Ben-Hamadou et al., "MIRACL-VC1: A Multi-Speaker Visual Corpus for Lip-Based Speaker Verification," IEEE
  International Conference on Acoustics, Speech and Signal Processing, 2014.
  [19] S. Petridis et al., "Audio-Visual Speech Recognition with a Hybrid CTC/Attention Architecture," 2018.
  [20] Z. Wu et al., "Lip Reading Sentences in the Wild," Machine Learning, 2016.
  [21] H. Kim, J. H. Hong, B. Roh, "Lip-to-Speech Synthesis in the Wild with Multi-Task Learning," KAIST, 2020.
  [22] S. Petridis, Y. Wang, Z. Li, M. Pantic, "End-to-End Audiovisual Fusion with LSTMs," IEEE Transactions on
  Affective
  Computing, 2019.
  [23] X. Yang et al., "LRW-1000: A Naturally-Distributed Large-Scale Benchmark for Lip Reading in the Wild," Pattern
  Recognition Letters, 2019.
  [24] A. Graves, S. Fernández, F. Gomez, and J. Schmidhuber, “Connectionist temporal classification: labelling
  unsegmented sequence data with recurrent neural networks,” in Proceedings of the 23rd international conference on
  Machine learning. ACM, 2006, pp. 369–376.
  [25] Zhao, Y., Xu, R., & Song, M. (2020). A cascade sequence-to-sequence model for Chinese Mandarin lip reading. In
  Proceedings of the 1st ACM International Conference on Multimedia in Asia (Article No. 32, pp. 1-6).
  ACM. https://doi.org/10.1145/3338533.3366579

## Appendix: