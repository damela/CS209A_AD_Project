---
nav_include: 4
title: Discussion and Conclusions
notebook: Conclusion.ipynb
---

## Discussion of the Results

This project applies data science to investigate and predict the progression of Alzeimer's Disease based on a patient's medical history. Using the Alzheimer’s Disease Neuroimaging Initiative (ADNI), a vast longitudinal and multicenter study, we have built data-driven models to predict current and future patient's mental health. In our analysis, we have considered two responses as indicators of the degree of dementia in a patient. The first one, the Clinical Dementia Rating (CDRSB), represents a numerical score out of $20$ given by the doctor to assess the illness magnitude. The second response is the diagnosis given by the doctor which falls into one of the following categories:

* Cognitively Normal (CN)
* Mild Cognitive Impairment (MCI)
* Alzheimer's Disease (AD)

The two responses can be approximately linked together as shown in the EDA through a simple classification function:

* CDRSB $< 0.5$: Diagnosis is CN
* $0.5 \le$ CDRSB$ < 4.0$: Diagnosis is MCI
* CDRSB$\ge 4.0$: Diagnosis is AD

In the first part of the project, rather than taking into account that patients in the study have associated data for multiple visits, we have considered each observation to be independent. This allowed us to investigate how well data-driven models can predict current CDRSB score and diagnosis based on the available predictors including demographics, medical history, and imaging data. While this approach might not be relevant to predict the progression of the disease, it gave us insights into the performance of different models as well as a ranking of the predictors' correlations. Indeed, ensemble models, such as Random Forest, performed well in predicting both the CDRSB score ($R^2$ above $87.6$%) and diagnosis (accuracy above $88.0$%). In addition, this first analysis revealed that cognitive tests such as the Functional Activities Questionnaire (FAQ) and Everyday Cognition tests (Ecog) were more correlated to the response variables than more expensive tests such as imaging data.

In the second part of the project, the longitudinal aspect of the database was taken into account to infer progression of AD. In order to predict future responses (CDRSB score and diagnosis), we first assumed a linear progression of the CDRSB score over time. This allowed us to engineer new predictors corresponding to the current CDRSB slope so far for each patient. Based on these new predictors, we further investigated the performance of the models based on two critical objectives in predicting AD: (1) early detection and (2) reduction of cost. This led to the creation of different dataframes with increasing number of predictors and data from an increasing number of follow-up visits. With sample weighting, to take into account that patients drop out of the study at different times, and hyper-parameter tuning we were able to find a promissing model that only uses data for the single $6$-month follow-up visit and a limited subset of predictors. This model has an $85.2$% accuracy on the diagnosis. Additionally, this model was shown to have relatively similar performance across the three diagnosis classes. Moreover, the model never misclassified AD as CN or vice versa.

One particularly desirable property of our model is that by predicting a patient's CDRSB trajectory as a function of time, we can make more nuanced estimates of their future mental health (compared to a simple classifier that only outputs one of the three categories: CN, MCI, and AD). In particular, the NIA has distinguished three sub-categories of AD (mild, moderate, and severe), and thus, our model could be used to extract this more detailed information [1].

In summary, based on the results obtained from data-drive models, here are our recommendation to predict accurately progression of AD:

* Use data from the first $6$-month visit along with an indicator of the slope of the CDRSB score since the baseline visit
* Use a limited subset of features including just demographics and cognitive tests

## Possible Improvements and Future Work 

Several improvements could be made to the presented models of this project. For instance, the results did not reveal significant impact of imaging data on the models' performance. This insight is somewhat at odds with the current literature where imaging data such at PET scan of the brain are used extensively to predict AD more and more reliably. The reason of the unimportance of imaging data in our project might come from the missing data associated with it. Features related to imaging are expensive and only a few patients had associated data. 

The confusion matrix presented in the modeling section also revealed a ~$18.2$% false-negative rate where AD patients were misclassified as MCI. In future models, one might want to penalize such false negatives due to their detrimental clinical implications.

The comparison between the models performance to predict current and future data has also exposed a challenge of the project. The responses (CDRSB score and diagnosis) were predicted based on another engineered response which corresponds to the current CDRSB slope. Therefore, the performance of the model in predicting the CDRSB score is based on the assumption that its progression is linear over time. It would be interesting to investigate other relations such as exponential and their potential impact on the results. Similarly, the final diagnosis was predicted based on an imputed classifier coming from the histogram of the CDRSB score by diagnosis. It would be interesting to study the sensitivity of the models performance with respect to the boundaries of this classifier.

### References

[1] NIA, Alzheimer's Disease Fact Sheet (2017). https://www.nia.nih.gov/health/alzheimers-disease-fact-sheet
