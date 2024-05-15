# Final Project: Text Classification on Toxic Comments

UC Berkeley MIDS Program 

Shuo Wang, Ci Song 

Spring 2024

Our project is heavily isnpired by the <a href="https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge">Jigsaw Toxic Comment Classification challenge</a> on Kaggle.

- [Final Project: Text Classification on Toxic Comments](#final-project-Text-Classification-on-Toxic-Comments)
  - [Project Overview](#project-overview)
  - [Dataset](#dataset)
  - [Models](#models)
  - [Experienment Results](#experienment-results)
  - [Model Evaluation and Metrics Summary](#model-evaluation-and-metrics-summary)
  - [Future Work](#future-work)
  - [Helpful Information](#helpful-information)
    - [Model Background](#model-background)
    - [Pydantic Model Expectations](#pydantic-model-expectations)
    - [Poetry Dependancies](#poetry-dependancies)
    - [Git Large File Storage (LFS)](#git-large-file-storage-lfs)
  - [Submission](#submission)
  - [Grading](#grading)
  - [Time Expectations](#time-expectations)

## Project Overview

In this research project, a banch of traditional machines learnings models and advanced Natural Language Processing (NLP) transformer models have been utilized to do the toxic comment classificaiton. Through meticulous experimentation on the Jigsaw Toxic Comment Classification dataset, it was
revealed that fine-tuned transformer-based models not only substantially improved accuracy, precision, recall, F1 score, and ROC AUC metrics but also demonstrated RoBERTa and DistilBERT’s slight superiority across nearly all metrics. A model evaluation and metrics summary table is at the end of the project. 

## Dataset
![alt text](https://github.com/Shuo-Wang-UCBerkeley/2024-spring-assignment-W266-NLP_Final_Project/blob/main/Images/Dataset.png)
## Models
- Baseline Models
    - CountVectorizer - Complement Naive Bayes (CNB)
    - CountVectorizer - Multinomial Naive Bayes (MNB)
    - TfidfVectorizer - CNB
    - TfidfVectorizer - MNB
- Transformer Models
    - BART
    - BERT
    - BERT+CNN
    - DistilBERT
    - DistilBERT+CNN
    - ALBERT
    - RoBERTa
    - Bidirectional_GRU
- Models
    - Deep Averaging Network (DAN)
        - DAN-Static
        - DAN_Retrain_word2vec
        - DAN-REtrain_uniform
    - Weighted Averaging Networks (WANs)
    - Logestic Regression
    - CNN
        - CNN-non_Retrain
        - CNN-Retrain
    - RNN
        - RNN-non_Retrain
        - RNN-Retrain
    - CNN_RNN
        - CNN_RNN-non_Retrain
        - CNN_RNN-Retrain
## Experienment Results
![alt text](https://github.com/Shuo-Wang-UCBerkeley/2024-spring-assignment-W266-NLP_Final_Project/blob/main/Images/Experiment_Results.png)

## Model Evaluation and Metrics Summary
![alt text](https://github.com/Shuo-Wang-UCBerkeley/2024-spring-assignment-W266-NLP_Final_Project/blob/main/Images/Model_Evaluation_Metrics_Summary_1.png)
![alt text](https://github.com/Shuo-Wang-UCBerkeley/2024-spring-assignment-W266-NLP_Final_Project/blob/main/Images/Model_Evaluation_Metrics_Summary_2.png)

## Future Work
1. Multi-Class Text Classification. 

    The current data is highly imbalanced, a potential solution is to build a two-step models.  

2. Parallel Computing Technique

    Utilizing parallel computing technique to work for the advanced NLP algorithms, including RoBERTa-LONG, T5, and XLNET. These 3 models resulted in Resource Exhausted Errors with the limited GPU capacity in Google Colab.

![alt text](https://github.com/Shuo-Wang-UCBerkeley/2024-spring-assignment-W266-NLP_Final_Project/blob/main/Images/Label_Frequency.png)