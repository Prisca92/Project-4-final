# Project-4-final

Machine Learning Integration   Heart Disease Risk Prediction



Project Description


Contributors:
Prisca Gossin,
Mehrin Khan 

Built With
* Python and Packages (eg. scikit-learn,seaborn, matplotlib,pandas,numpy, etc.)
* Tableau


Overview:

Our objective is to create a machine learning model for detecting heart disease by focusing on various causative attributes, aiming to predict early signs of heart disease to identify individuals at risk. Analyzing relevant data allows us to anticipate and catch potential issues early on based on specific variables or features, improving health outcomes and ensuring efficient use of healthcare resources. Health care professionals will be able to provide recommendations for managing and reducing the risk of heart disease, including lifestyle modifications and preventive measures, to improve overall health and minimize disease risk for at-risk patients.




Questions to Investigate: 


What are the key features or attributes suspected to be associated with heart disease?


What geographical factors contribute to being at risk of heart disease?


Are there any patterns or relationships between the features and the target variable (heart disease)?


Is there any connection between age, cholesterol, and chest paint type when it comes to heart disease?
                                  
                                  
                                  
                                  
                                  
                                  
                                  
                MACHINE LEARNING TRAINING MODEL PROCESS

a) Data Pre-processing
b) Exploratory Data Analysis, 
c) Outlier Detection & Removal, 
d) Training and Test Split, 
e) Cross Validation, 
f) Model Building, 
g) Model evaluation & comparison, 
h) Feature Selection, 
i) Model Evaluation.
                    
                     
                     
                     DATA COLLECTION/DATA PRE-PROCESSING

Data pre-processing is an essential step in machine learning to represent data suitable for the algorithms and classification. Data collection is simply reading the data, specifying the type of variables, as well as which variable is the target to predict.



Data was collected from Kaggle: 

Target variable: Heart disease 


Original Data

<img width="788" alt="Screenshot 2024-02-10 at 11 53 05 AM" src="https://github.com/Prisca92/Project-4-final/assets/140542648/65335be8-5fe5-4888-bb62-a1a3d2c2264c">



Feature_Description

<img width="613" alt="Screenshot 2024-02-13 at 4 40 36 AM" src="https://github.com/Prisca92/Project-4-final/assets/140542648/a0a58ceb-7a72-4df5-8058-3a569b8a8d54">


                        
                        
                                                        EXPLORATORY DATA ANALYSIS (EDA)
                        

This step was used to make the original data ready for model building. This Data pre-processing step included variable scaling, converting categorical variable to numerical. In general, it is a series of tasks to make original data consumable and understandable for machine-learning models. 

Processing:
- Normalizing the data
- Renaming columns
- Transforming columns to be numerical features
- using pd.get_dummies to convert categorical data to numerical features
- Converting file to csv file
- Check for balance




Data Ditribution using Histogram graph


<img width="591" alt="Screenshot 2024-02-12 at 8 57 29 PM" src="https://github.com/Prisca92/Project-4-final/assets/140542648/421c986d-320d-42dc-afe7-a856cd5113cc">


Search for Missing Values


<img width="528" alt="Screenshot 2024-02-12 at 8 58 15 PM" src="https://github.com/Prisca92/Project-4-final/assets/140542648/3a8c372f-d5ad-46ca-8c6c-e7b203bde925">


Examine unique values


<img width="437" alt="Screenshot 2024-02-12 at 8 56 46 PM" src="https://github.com/Prisca92/Project-4-final/assets/140542648/9090a855-d230-4745-aac3-f6e7ab7a0a6a">



Normalizing data using git.dummies


<img width="986" alt="Screenshot 2024-02-13 at 12 54 20 PM" src="https://github.com/Prisca92/Project-4-final/assets/140542648/6a3a50e1-4b9b-4eaa-9de1-c85e7e523b2b">




Balanced dataset

<img width="310" alt="Screenshot 2024-02-13 at 8 09 46 PM" src="https://github.com/Prisca92/Project-4-final/assets/140542648/6aa70a11-8077-475a-b2c7-551e1c2db5ee">

                  



Train and Test model




Cross Validation



                     
                     
                     
                                                   MODEL BUILDING/SELECTION


Supervised Learning/ RandomForestClassifier


The model building process finds the model that fits best for the training data set in terms of prediction accuracy. One of the most popular approaches to achieve this goal is to iterate over multiple related machine learning models to see which one is the best fit. For this project we opted for a supervised random forest model to address our task. Supervised learning was our preference because we have labeled data, where features are organized in a tabular format alongside corresponding outputs. This setup allows us to train the model to make predictions based on input features. We focused on target variables, also known as dependent variables, which are the outcomes we aim to predict(heart disease). In our scenario, our target variable is the extent to which each feature contributes to the risk of heart disease. Three regression model classes were tested – RandomForest, DecisionTree and Linear Regression – ultemitely we decided to use the one with highest r-square  as the best fit.  Both RandomforestClassifer and the linear regression model provided an accuracy score of 90%. Ultimately RandomforestClassifier was the training model we choose as the best fit as it is the most reliable an accurate model due to its high accuracy, feature important and versatility. In the Medical sector when making predictions it is best to use the most reliable machine learning model for prediction purposes. This system ultimately can help suggest precautions to the user based on output. 



                                EVALUATION
   
    
Key insights when evelautaing the Randomforest model's effectiveness in its classification task.   

To better evaluate the random forest we used confusion matrix  which is a table that shows how many instances of each class were correctly or incorrectly predicted. Helps identify source errors like false positives and false negatives which we can see below:



Confusion Matrix



<img width="530" alt="Screenshot 2024-02-13 at 4 45 14 AM" src="https://github.com/Prisca92/Project-4-final/assets/140542648/8c93e7ca-32a7-4c72-8db1-1f630200f718">


 



Feature Importance



<img width="720" alt="Screenshot 2024-02-13 at 4 45 40 AM" src="https://github.com/Prisca92/Project-4-final/assets/140542648/877a6141-ebc6-4688-8149-846967a0c2db">



Classification_Report


<img width="513" alt="Screenshot 2024-02-13 at 12 52 35 PM" src="https://github.com/Prisca92/Project-4-final/assets/140542648/85f9d404-a2b0-4875-b5bf-895ce65ade87">




This classification report provides a comprehensive evaluation of a binary classification model's performance.


Analysis of each metric:

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
Overall, this classification report suggests that the model performs well in distinguishing between the two classes, with high precision, recall, and F1-score values. The high accuracy and balanced performance metrics indicate that the model is effective in its classification task.Over all superior performance.



-We explored coorelation between individual features and the target variable can provide initial insights into potential predictors of heart disease.



Heart Disease vs Cholesterol Correlation

<img width="482" alt="Screenshot 2024-02-13 at 12 13 47 PM" src="https://github.com/Prisca92/Project-4-final/assets/140542648/16bc86c2-33f8-415d-bf53-2d19fea17629">


Heart Disease vs Age 


<img width="484" alt="Screenshot 2024-02-13 at 6 27 32 PM" src="https://github.com/Prisca92/Project-4-final/assets/140542648/7174e18d-94ba-4ce2-a44d-9943f9b0c7b6">


For more details see our final model


Tableau Link: [
](https://public.tableau.com/authoring/HeartDisease_17074856563830/Story1#1)        
<img width="612" alt="Screenshot 2024-02-13 at 7 03 19 PM" src="https://github.com/Prisca92/Project-4-final/assets/140542648/8eb49e85-7ce8-4b70-8a8d-9fdef5c69935">
                        
                         
                         
                         CONCLUSION


Limitations:
                         


Things to consider for further analysis:

- Dive in deeper into feature importance to re-evaluate the performance of the random forest model to identify interactions between features.
 - Focus on a specific region or compare certain regions this would allow for further geographical analysis in relations to age,sex,resting blood pressure(environmental factors such as altitude,temperature,diet, physical activity patterns,stress),cholesterol(dietary habits),fasting blood sugar(geographical variationin diet, physical activity),max heart rate(altitude and temp), and excersice induced angina(air quality),oldpeak(socioeconomic factors). Understanding how these geographical factors interact with individual characteristics and lifestyle behaviors can provide insights into regional disparities in heart disease risk and guide targeted interventions and public health initiatives.
 - 
<img width="425" alt="Screenshot 2024-02-13 at 1 45 38 PM" src="https://github.com/Prisca92/Project-4-final/assets/140542648/a447217e-a50f-477f-b6dd-cc6aa3c3e50c"> 
 

