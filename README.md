# Final Project: Text Classification on Toxic Comments

UC Berkeley MIDS Program 

Shuo Wang, Ci Song 

Spring 2024

Our project is heavily isnpired by the <a href="https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge">Jigsaw Toxic Comment Classification challenge</a> on Kaggle.

<p align="center">
    <!--Hugging Face-->
        <img src="https://user-images.githubusercontent.com/1393562/197941700-78283534-4e68-4429-bf94-dce7ab43a941.svg" width=7%>
    <!--PLUS SIGN-->
        <img src="https://user-images.githubusercontent.com/1393562/190876627-da2d09cb-5ca0-4480-8eb8-830bdc0ddf64.svg" width=7%>
    <!--FAST API-->
        <img src="https://user-images.githubusercontent.com/1393562/190876570-16dff98d-ccea-4a57-86ef-a161539074d6.svg" width=7%>
    <!--PLUS SIGN-->
        <img src="https://user-images.githubusercontent.com/1393562/190876627-da2d09cb-5ca0-4480-8eb8-830bdc0ddf64.svg" width=7%>
    <!--REDIS LOGO-->
        <img src="https://user-images.githubusercontent.com/1393562/190876644-501591b7-809b-469f-b039-bb1a287ed36f.svg" width=7%>
    <!--PLUS SIGN-->
        <img src="https://user-images.githubusercontent.com/1393562/190876627-da2d09cb-5ca0-4480-8eb8-830bdc0ddf64.svg" width=7%>
    <!--KUBERNETES-->
        <img src="https://user-images.githubusercontent.com/1393562/190876683-9c9d4f44-b9b2-46f0-a631-308e5a079847.svg" width=7%>
    <!--PLUS SIGN-->
        <img src="https://user-images.githubusercontent.com/1393562/190876627-da2d09cb-5ca0-4480-8eb8-830bdc0ddf64.svg" width=7%>
    <!--Azure-->
        <img src="https://user-images.githubusercontent.com/1393562/192114198-ac03d0ef-7fb7-4c12-aba6-2ee37fc2dcc8.svg" width=7%>
    <!--PLUS SIGN-->
        <img src="https://user-images.githubusercontent.com/1393562/190876627-da2d09cb-5ca0-4480-8eb8-830bdc0ddf64.svg" width=7%>
    <!--K6-->
        <img src="https://user-images.githubusercontent.com/1393562/197683208-7a531396-6cf2-4703-8037-26e29935fc1a.svg" width=7%>
    <!--PLUS SIGN-->
        <img src="https://user-images.githubusercontent.com/1393562/190876627-da2d09cb-5ca0-4480-8eb8-830bdc0ddf64.svg" width=7%>
    <!--GRAFANA-->
        <img src="https://user-images.githubusercontent.com/1393562/197682977-ff2ffb72-cd96-4f92-94d9-2624e29098ee.svg" width=7%>
</p>

- [Final Project: Text Classification on Toxic Comments](#final-project-Text-Classification-on-Toxic-Comments)
  - [Project Overview](#project-overview)
  - [Dataset](#dataset)
  - [Models](#models)
  - [Experienment Results](#experienment-results)
  - [Model Evaluation and Metrics Summary](#model-evaluation-and-metrics-summary)
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