# Project-4-final

Project Description

Contributors:
Prisca Gossin,
Mehrin Khan 

Built With
* Python and Packages (eg. scikit-learn,seaborn, matplotlib,etc.)
* Tableau


Overview:

Our objective is to create a machine learning model for detecting heart disease by focusing on various causative attributes, aiming to predict early signs of heart disease to identify individuals at risk. Analyzing relevant data allows us to anticipate and catch potential issues early on based on specific variables or features, improving health outcomes and ensuring efficient use of healthcare resources. Health care professionals will be able to provide recommendations for managing and reducing the risk of heart disease, including lifestyle modifications and preventive measures, to improve overall health and minimize disease risk for at-risk patients.


Factors:

Target variable: Heart disease 


Questions to investigate: 


What are the key features or attributes suspected to be associated with heart disease?

what geographical factors contribute to being at risk of heart disease?


Are there any patterns or relationships between the features and the target variable (heart disease)?

we explored coorelation between individual features and the target variable can provide initial insights into potential predictors of heart disease.

Is there any connection between age, cholesterol, and chest paint type when it comes to heart disease?
                                  
                                  
                                  
                                  
                                  
                                  
                                  
                MACHINE LEARNING TRAINING MODEL PROCESS



                     DATA COLLECTION/READING DATA

This is simply reading the data, specifying the type of variables, as well as which variable is the target to predict.

Data was collected from Kaggle: 

snippet of code:

screenshot showing feature description:



                        DATA PRE-PREOCESSING

This step was used to make the original data ready for model building. This Data pre-processing step included data imputation, variable scaling, converting categorical variable to numerical. In general, it is a series of tasks to make original data consumable and understandable for machine-learning models. 

We processed the data by:
-Normalizing the data
-Renaming columns
-Transforming columns to be numeric
-using pd.get_dummies() to convert categorical data to numeric
-Converting orginal data which was in an excel sheet to a csv file

snippet normalizing data





                     MODEL BUILDING AND SELECTION

Supervised learning:

The model building process finds the model that fits best for the training data set in terms of prediction accuracy. One of the most popular approaches to achieve this goal is to iterate over multiple related machine learning models to see which one is the best fit. For this project we opted for a supervised random forest regression model to address our task. Supervised learning was our preference because we have labeled data, where features are organized in a tabular format alongside corresponding outputs. This setup allows us to train the model to make predictions based on input features. We focused on target variables, also known as dependent variables, which are the outcomes we aim to predict(at-risk for heart disease). In our scenario, our target variable is the extent to which each feature contributes to the risk of heart disease.Wused three regression model classes – RandomForest, DecisionTree and Linear Regression – are fitted, and the one with highest r-square is picked as the best fit.  Both Randomforest and the linear regression model provided an accuracy score of 90%. Ultimately Randomforest classifier was the training model we choose as the best fit as it is the most reliable an accurate model due to its high accuracy, feature important and versatility. In the Medical sector when making predictions it is best to use the most reliable machine learning model for prediction purposes. This system ultimately help suggest precautions to the user based on output. To better evaluate the random forest we used confusion matrix  which is a table that shows how many instances of each class were correctly or incorrectly predicted.Helps identify source errors like false positives and false negatives which we did not encounter. 


random classifier code snippet:


confusion matix snippet:




                                EVALUATION
   
    
Key insights when evelautaing the Randomforest model's effectiveness in its classification task.
    
    Model 1: Over all superior performance.
    
    
    
    Add classification model snippet:


This classification report provides a comprehensive evaluation of a binary classification model's performance. Here's the analysis of each metric:

Precision:

Precision measures the proportion of true positive predictions among all positive predictions made by the model.
For class 0 (presumably negative instances), the precision is 0.88, indicating that 88% of the instances predicted as negative are indeed negative.
For class 1 (presumably positive instances), the precision is 0.92, indicating that 92% of the instances predicted as positive are indeed positive.
Overall, the model has high precision for both classes, suggesting that it makes relatively few false positive predictions.
Recall:

Recall measures the proportion of true positive predictions among all actual positive instances in the dataset.
For class 0, the recall is 0.88, indicating that 88% of the actual negative instances are correctly identified by the model.
For class 1, the recall is 0.92, indicating that 92% of the actual positive instances are correctly identified by the model.
The model exhibits high recall for both classes, suggesting that it captures a large portion of the positive instances in the dataset.
F1-score:

The F1-score is the harmonic mean of precision and recall, providing a balanced measure of a model's performance.
For class 0, the F1-score is 0.88, indicating a balanced performance between precision and recall for negative instances.
For class 1, the F1-score is 0.92, indicating a balanced performance between precision and recall for positive instances.
The weighted average F1-score is also 0.90, suggesting a good overall balance between precision and recall across both classes.
Support:

Support refers to the number of actual occurrences of each class in the dataset.
There are 74 instances of class 0 and 110 instances of class 1 in the dataset.
Accuracy:

Accuracy measures the overall correctness of the model's predictions, regardless of class.
The model achieves an accuracy of 0.90, indicating that it correctly predicts the class label for 90% of the instances in the dataset.
Overall, this classification report suggests that the model performs well in distinguishing between the two classes, with high precision, recall, and F1-score values. The high accuracy and balanced performance metrics indicate that the model is effective in its classification task.

See final model:



Tableau Link: 
        
        
                         CONCLUSION
                         
Things to consider for further analysis

- Dive in deeper into feature importance to re-evaluate the performance of the random forest model to identify interactions between features.
 - Focus on a specific region or compare certain regions this would allow for further geographical analysis in relations to age,sex,resting blood pressure(environmental factors such as altitude,temperature,diet, physical activity patterns,stress),cholesterol(dietary habits),fasting blood sugar(geographical variationin diet, physical activity),max heart rate(altitude and temp), and excersice induced angina(air quality),oldpeak(socioeconomic factors). Understanding how these geographical factors interact with individual characteristics and lifestyle behaviors can provide insights into regional disparities in heart disease risk and guide targeted interventions and public health initiatives.
 

