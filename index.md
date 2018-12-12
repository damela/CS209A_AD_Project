---
title: Alzheimer's Disease and Cognitive Impairment Prediction
---

## CS209A Final Project

>Group $39$: Connor McCann, David Melancon, Zixi Liu

## Project Statement

Alzheimer's disease (AD) is a neurodegenerative disease associated with slow decrease of mental functions. Most commonly, the disease appears after $65$ years of age and affect about $6$% of people. The causes of AD are still not well understood and no treatment has yet been discovered. As such, early diagnosis is critical to aid in slowing the progression of the disease. In addition to the human impact it provokes, AD is also one of the most costly dieases in the United States due to the long-term care it requires.

Longitudinal multicenter investigations, such as the Alzheimer's Disease Neuroimaging Initiative (ADNI), are bringing a paradigm shift in the analysis of AD. These studies enable the generation of large databases of patient information such as demographics, medical history, lab records, cognitive test scores, and imaging data. In this project, based on the ADNI database, we aim to develop a data-driven model to predict AD at an early stage and identify the most cost efficient tests to make such predictions.

## Project Goals

The project's overarching goal is to predict over time if a patient will develop AD. We are especially concerned about accurate, fast, and inexpensive detection. The goals can be formulated as follows:

1. Based on data at the first visit, predict the progression of the disease
2. Investigate the accuracy of the model as a function of the number of visits included in the model
3. Investigate the accuracy of the model as a function of the feature selection