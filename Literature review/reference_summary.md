# Summary of reference


## (S. Fenghour, D. Chen, K. Guo, B. Li and P. Xiao, 2021, "Deep Learning-Based Automated Lip-Reading: A Survey,")
**a servey for lip reading** [Deep_Learning-Based_Automated_Lip-Reading_A_Survey.pdf](..%2Frefference%2FDeep_Learning-Based_Automated_Lip-Reading_A_Survey.pdf)


### The survey topic: 
The paper is a survey on automated lip-reading approaches that use deep learning methods for feature extraction and classification. It also provides comparisons of different components and challenges of lip-reading systems.
### The survey contributions:
The paper claims to have some unique insights, such as comparing different neural network architectures, reviewing the advantages of attention-transformers and temporal convolutional networks, and covering the most up-to-date lip-reading systems until early 2021.
### The survey motivation: 
The paper states that lip-reading is a multifaceted discipline that has evolved from recognizing isolated speech units to decoding natural spoken language1. It also states that lip-reading has potential applications in various domains such as security, entertainment, and education.
### Different types of CNNs for feature extraction: 
The article compares the advantages and disadvantages of using 2D, 3D, or 2D+3D convolutional neural networks (CNNs) for extracting spatial and temporal features from lip images. It also reviews some of the existing architectures that use CNNs for the frontend of lip-reading systems.
### RNNs, Attention, CTCs, and Transformers for classification: 
The article discusses the use of recurrent neural networks (RNNs), attention mechanisms, connectionist temporal classification (CTC), and transformers for the backend of lip-reading systems. It explains how these methods can handle sequence prediction, temporal alignment, and long-term dependencies. It also mentions some of the recent works that use these methods for lip-reading sentences.
### TCNs as an alternative to RNNs: 
The article introduces temporal convolutional networks (TCNs) as another option for sequence classification. It highlights the benefits of TCNs over RNNs, such as parallel computation, flexible receptive field size, and lower memory requirement. It also cites some of the works that use TCNs for lip-reading words.
### Classification schema for lip-reading: 
The article explores the different ways of encoding speech units for lip-reading, such as words, ASCII characters, visemes, and phonemes. It analyzes the pros and cons of each schema, and how they affect the performance and generalization of lip-reading systems.
### Audio-visual speech recognition (AVSR) datasets:
* AVLetters, AVICAR, Tulips, M2VTS, and AVDigits for letter and digit recognition.
* MIRACL-VC1, LRW, LRW-1000, IBMViaVoice, OuluVS1, OuluVS2, GRID, LRS2, MV-LRS, LRS3-TED, and LSVSR for word and sentence recognition.
* CUAVE, MIRACL-VC1, OuluVS2, MV-LRS, and HIT-AVDB-II for multiview recognition.

The paper also provides a brief description of each dataset, such as the number of speakers, words, sentences, views, languages, and resolutions. The paper also compares the datasets in terms of their challenges and applications.
### Deep learning methods for lip reading: 
A review of recent advances in applying deep neural networks, such as convolutional neural networks (CNNs), recurrent neural networks (RNNs), long short-term memory (LSTM) networks, and attention mechanisms, to lip reading tasks, such as word or sentence recognition .
### Challenges and future directions for lip reading:
the challenges and potential advancements in automated lip-reading systems. It mentions the possibility of creating lexicon-free systems that could predict words not present in the training phase by using phonemes and visemes. However, several challenges remain, such as predicting unseen words, handling visual ambiguities, and generalizing to speakers not included in the training data. Additionally, the systems need to adapt to videos of varying spatial resolution and different frame rates. These factors contribute to the complexity of developing effective automated lip-reading systems.

## (Y. Assael, B. Shillingford, S. Whiteson and N. de Freitas, 2016, "LipNet: End-to-End Sentence-level Lipreading,")

### background:
The paper proposes a deep learning model that can map sequences of video frames of a speaker’s mouth to text sentences, without requiring any alignment or segmentation of the input. The model uses spatiotemporal convolutions, recurrent neural networks, and connectionist temporal classification loss, trained end-to-end.

### Dataset and Features:
The paper evaluates the model on the GRID corpus, a sentence-level lipreading dataset with a simple grammar and a limited vocabulary. The paper also uses data augmentation techniques such as mirroring, word clipping, and frame deletion and duplication.

### methods and results:
The paper reports that the model achieves 95.2% accuracy in sentence-level word prediction, outperforming experienced human lipreaders and the previous word-level state-of-the-art accuracy of 86.4%. The paper also shows that the model can generalize across unseen speakers and attends to phonologically important regions in the video.

### conclusion:
The paper claims that the model is the first to apply deep learning to end-to-end learning of a model that maps sequences of image frames of a speaker’s mouth to entire sentences. The paper also suggests that the model can be improved with more data and applied to larger datasets and audio-visual speech recognition tasks.
## (J, S. Chung,  A. Senior, O, Vinyals,A, Zisserman, 2017, "Lip Reading Sentences in the Wild")

### background and abstract:
The paper provide a novel network model for audio-visual speech recognition that can transcribe speech into characters from visual input only, audio input only, or both. The model uses a dual attention mechanism to align the input sequences and the output characters. The model is called Watch, Listen, Attend and Spell (WLAS).

### Dataset and Features:
The paper  use a  new large-scale dataset for lip reading, consisting of over 100,000 natural sentences from British television. The dataset is called Lip Reading Sentences (LRS). The dataset contains a wide variety of speakers, poses, expressions, lighting, and backgrounds.

### Methods:

The paper use a training strategy that uses curriculum learning, scheduled sampling, multi-modal training, and noisy audio training to improve the performance and generalization of the model.

### Results and Discussion:
The paper presents an evaluation of the model on the LRS dataset and two public benchmark datasets for lip reading: LRW and GRID. The model achieves state-of-the-art results on all datasets, and surpasses the performance of a professional lip reader on the LRS dataset. The model also demonstrates that visual information helps to improve speech recognition performance even when the audio is available.

### Conclusion and Future Work:

The paper acknowledges that the model is limited by the quality and quantity of the training data, and that lip reading is inherently ambiguous due to homophemes and co-articulation. The paper also notes that the model is trained and tested in batch mode, which is different from how humans lip read in real time. The paper suggests that incorporating monotonicity constraints, using an online architecture, and discerning important discriminative cues could improve the model’s performance and robustness.
Potential applications and impacts: The paper discusses several possible applications of lip reading, such as dictating instructions or messages in a noisy environment, transcribing and re-dubbing archival silent films, resolving multi-talker simultaneous speech, and improving the performance of automated speech recognition in general3. The paper also speculates that lip reading research could benefit the hard of hearing by teaching them how to lip read better.


## (A,Gutierrez,Z A,Robert, 2017,"Lip Reading Word Classification")
### Dataset and Features: 
The authors use the MIRACL-VC1 dataset, which contains color and depth images of 15 speakers uttering 10 words and 10 phrases. They preprocess the data by cropping the faces and augmenting the data with flipping and jittering. They focus on the word classification task with 10 classes.

### Methods: 
The authors describe four models: a Baseline CNN + LSTM network, a Deep Layered CNN + LSTM network inspired by LipNet, an LSTM network on top of VGG-16 features pretrained on ImageNet, and a Fine-Tuned VGG-16 + LSTM network34. They explain the rationale and structure of each model and the techniques they use, such as dropout, batch normalization, and transfer learning.

### Results and Discussion: 
The authors report the accuracy of their models on both seen and unseen subjects. They find that the Fine-Tuned VGG-16 + LSTM network achieves the best performance, with 79% validation accuracy and 59% test accuracy on seen subjects5. They also analyze the confusion matrices of their models and identify some sources of errors, such as short words and speaker variations. They suggest some ways to improve their models, such as regularization, cross-validation, and data augmentation.

## (T,Afouras,J,S Chung,A,Zisserman, 2018,"Deep Lip Reading: a comparison of models and an online application")

### Background and Abstract:
The paper presents a comparison of different deep learning models for lip reading, including LipNet, Lip Reading Sentences (LRS), and Lip Reading Words (LRW). The paper also describes an online lip reading application that uses the best performing model.

### Dataset and Features:
The paper uses the LRS dataset, which contains 500 videos of 34 speakers uttering 10 sentences each. The paper also uses the LRW dataset, which contains 500 videos of 500 speakers uttering 1000 words each. The paper preprocesses the data by cropping the faces and augmenting the data with flipping and jittering.

### Methods:
The paper describes the architecture of the models and the techniques used to train them, such as dropout, batch normalization, and transfer learning. The paper also explains the rationale behind the models and the differences between them.

### Results and Discussion:
The paper reports the accuracy of the models on the LRS and LRW datasets. The paper finds that the LRS model achieves 82.4% accuracy on the LRS dataset and 65.4% accuracy on the LRW dataset. The paper also finds that the LRW model achieves 87.2% accuracy on the LRW dataset and 65.4% accuracy on the LRS dataset. The paper also analyzes the confusion matrices of the models and identifies some sources of errors, such as short words and speaker variations. The paper suggests some ways to improve the models, such as regularization, cross-validation, and data augmentation.

### Conclusion and Future Work:
The paper concludes that the LRS model is better suited for sentence-level lip reading, while the LRW model is better suited for word-level lip reading. The paper also suggests that the models could be improved by using more data, better preprocessing, and better training techniques.

### Potential applications and impacts:
The paper discusses several possible applications of lip reading, such as dictating instructions or messages in a noisy environment, transcribing and re-dubbing archival silent films, resolving multi-talker simultaneous speech, and improving the performance of automated speech recognition in general3. The paper also speculates that lip reading research could benefit the hard of hearing by teaching them how to lip read better.