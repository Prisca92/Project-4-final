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

Our objective is to create a machine learning model for detecting heart disease by focusing on various causative attributes, aiming to predict early signs to identify individuals at risk. Analyzing relevant data allows us to anticipate and catch potential issues early on based on specific variables or features, improving health outcomes and ensuring efficient use of healthcare resources. Health care professionals will be able to provide recommendations for managing and reducing the risk of heart disease, including lifestyle modifications and preventive measures, to improve overall health and minimize disease risk for at-risk patients.




Questions to Investigate: 


1. What are the key features or attributes suspected to be associated with heart disease?
2. What geographical factors contribute to the risk of heart disease?
3. Are there any patterns or relationships between the features and the target variable (heart disease)?

                                  
                                  
                                  
                                  
                                  
                MACHINE LEARNING TRAINING MODEL PROCESS

a) Data Pre-processing
b) Exploratory Data Analysis, 
c) Training and Test Split, 
d)Outlier Detection
e) Model Building,  
f) Feature Selection, 
g) Model Evaluation.
                    
                     
                     
                     DATA COLLECTION/DATA PRE-PROCESSING

Data preprocessing is a vital step in machine learning, essential for preparing data to be suitable for algorithms and classification tasks. Data collection involves reading the data, identifying variable types, and determining the target variable for prediction.

Data was collected from Kaggle:  https://www.kaggle.com/fedesoriano/heart-failure-prediction


This dataset is a sample created by pre-existing combined data to focus on specific features. 
In this dataset, 5 heart datasets are combined over 11 common features. The five datasets used for it's curation derived from:

Observations:
Cleveland,
Hungarian,
Switzerland,
Long Beach VA,
Stalog (Heart) Data Set: 
Total: 1190 observations
Duplicated: 272 observations




Target variable: Heart disease 


Original Data

<img width="788" alt="Screenshot 2024-02-10 at 11 53 05 AM" src="https://github.com/Prisca92/Project-4-final/assets/140542648/65335be8-5fe5-4888-bb62-a1a3d2c2264c">



Feature Description

<img width="613" alt="Screenshot 2024-02-13 at 4 40 36 AM" src="https://github.com/Prisca92/Project-4-final/assets/140542648/a0a58ceb-7a72-4df5-8058-3a569b8a8d54">


                        
                        
                            EXPLORATORY DATA ANALYSIS (EDA)
                        

This step was used to make the original data ready for model building. In general, it is a series of tasks to make original data consumable and understandable for machine-learning models. 

Processing:
- Normalizing the data
- Adding columns
- Convert categorical data to numerical features
- Converting file to csv file
- Check balance




Data Ditribution using Histogram graph


<img width="591" alt="Screenshot 2024-02-12 at 8 57 29 PM" src="https://github.com/Prisca92/Project-4-final/assets/140542648/421c986d-320d-42dc-afe7-a856cd5113cc">


Examine for missing values: None


<img width="528" alt="Screenshot 2024-02-12 at 8 58 15 PM" src="https://github.com/Prisca92/Project-4-final/assets/140542648/3a8c372f-d5ad-46ca-8c6c-e7b203bde925">




Data Types



<img width="242" alt="Screenshot 2024-02-13 at 8 56 12 PM" src="https://github.com/Prisca92/Project-4-final/assets/140542648/2feab8b0-2e27-46d9-9254-a3a4d72aca42">




Examine unique values


<img width="437" alt="Screenshot 2024-02-12 at 8 56 46 PM" src="https://github.com/Prisca92/Project-4-final/assets/140542648/9090a855-d230-4745-aac3-f6e7ab7a0a6a">



Normalizing data 


<img width="986" alt="Screenshot 2024-02-13 at 12 54 20 PM" src="https://github.com/Prisca92/Project-4-final/assets/140542648/6a3a50e1-4b9b-4eaa-9de1-c85e7e523b2b">




Balanced dataset: Yes

<img width="310" alt="Screenshot 2024-02-13 at 8 09 46 PM" src="https://github.com/Prisca92/Project-4-final/assets/140542648/6aa70a11-8077-475a-b2c7-551e1c2db5ee">




Heart Disease vs Max Heart Rate


<img width="452" alt="Screenshot 2024-02-14 at 6 15 40 PM" src="https://github.com/Prisca92/Project-4-final/assets/140542648/4064cda2-f67d-4645-8125-ad885710a7c8">

                  




Correlation of features 


<img width="1014" alt="Screenshot 2024-02-14 at 2 57 28 PM" src="https://github.com/Prisca92/Project-4-final/assets/140542648/75bc9dbb-c7ea-43ef-9e75-7b9d265b78e3">

High positive correlations indicate a strong positive linear relationship, while high negative correlations indicate a strong negative linear relationship. Correlations close to zero suggest little to no linear relationship.




-We explored correlation between individual features and the target variable can provide initial insights into potential predictors of heart disease.



Heart Disease vs Cholesterol Correlation


<img width="529" alt="Screenshot 2024-02-14 at 3 03 57 PM" src="https://github.com/Prisca92/Project-4-final/assets/140542648/15937dc1-e61c-49ab-bf67-85e1fbb6002c">

Heart Disease vs Max Heart Rate


<img width="452" alt="Screenshot 2024-02-14 at 6 15 40 PM" src="https://github.com/Prisca92/Project-4-final/assets/140542648/4064cda2-f67d-4645-8125-ad885710a7c8">


                     
                     
                       MODEL BUILDING/SELECTION


Supervised Learning/ Random Forest 


The model building process finds the model that fits best for the training data set in terms of prediction accuracy. One of the most popular approaches to achieve this goal is to iterate over multiple related machine learning models to see which one is the best fit. For this project three regression model classes were tested – RandomForest, DecisionTree and Linear Regression – ultimately we decided to use the one with highest accuracy score as the best fit to address our task. Supervised learning was our preference because we have labeled data, where features are organized in a tabular format alongside corresponding outputs. This setup allows us to train the model to make predictions based on input features. We focused on target variables, also known as dependent variables, which are the outcomes we aim to predict(heart disease). In our scenario, our target variable is the extent to which each feature contributes to the risk of heart disease.  Both RandomforestClassifer and the linear regression model provided an accuracy score of 90%. Ultimately RandomforestClassifier was the training model we choose as the best fit as it is the most reliable an accurate model due to its high accuracy, feature important and versatility. In the Medical sector when making predictions it is best to use the most reliable machine learning model for prediction purposes. This system ultimately can help suggest precautions to the user based on output. 



                                EVALUATION
   
    
Key insights when evaluating the Random forest model's effectiveness in its classification task.   

To better evaluate the random forest we used confusion matrix  which is a table that shows how many instances of each class were correctly or incorrectly predicted. Helps identify source errors like false positives and false negatives which we can see below:



Confusion Matrix



<img width="530" alt="Screenshot 2024-02-13 at 4 45 14 AM" src="https://github.com/Prisca92/Project-4-final/assets/140542648/8c93e7ca-32a7-4c72-8db1-1f630200f718">


Classification Report


<img width="513" alt="Screenshot 2024-02-13 at 12 52 35 PM" src="https://github.com/Prisca92/Project-4-final/assets/140542648/85f9d404-a2b0-4875-b5bf-895ce65ade87">




This classification report provides a comprehensive evaluation of a classification model's performance.


Analysis of each metric:

Precision:

Precision measures the proportion of true positive predictions among all positive predictions made by the model.
For class 0,(negative instances) the precision is 0.88, indicating that 88% of the instances predicted as negative are indeed negative.
For class 1, ( positive instances), the precision is 0.92, indicating that 92% of the instances predicted as positive are indeed positive.
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


Feature Importances



<img width="720" alt="Screenshot 2024-02-13 at 4 45 40 AM" src="https://github.com/Prisca92/Project-4-final/assets/140542648/877a6141-ebc6-4688-8149-846967a0c2db">



For more details see our Final Model File


Tableau Link:

Story1
https://public.tableau.com/views/HeartDisease_17074856563830/Story1?:language=en-US&:sid=&:display_count=n&:origin=viz_share_link


<img width="612" alt="Screenshot 2024-02-13 at 7 03 19 PM" src="https://github.com/Prisca92/Project-4-final/assets/140542648/8eb49e85-7ce8-4b70-8a8d-9fdef5c69935">
                        
                         
                         
                         CONCLUSION




Despite having a low false positive rate. We believe when making predictions it is more advantageous to falsely identify patients with heart disease. Having a High false negative rate would be more concerning as the aim of this model is to not only identify patients at risk but to also ensure healthcare professionals can give  recommendations for managing and reducing the risk of heart disease, including lifestyle modifications and preventive measures, to improve overall health and minimize disease risk for at-risk patients.



Limitations:
                         


Things to consider for further analysis:
Yes, Further analysis could have been conducted
-Not limiting ourselves to an already sampled dataset, using all the data available from the original data collected and combine it based on additional features.
- Dive in deeper into feature importance to re-evaluate the performance of the random forest model to identify interactions between features.
- Focus on a specific region or compare certain regions this would allow for further geographical analysis in relations to age,sex,resting blood pressure(environmental factors such as altitude,temperature,diet, physical activity patterns,stress),cholesterol(dietary habits),fasting blood sugar(geographical variationin diet, physical activity),max heart rate(altitude and temp), and excersice induced angina(air quality),oldpeak(socioeconomic factors). Understanding how these geographical factors interact with individual characteristics and lifestyle behaviors can provide insights into regional disparities in heart disease risk and guide targeted interventions and public health initiatives.
 - 
<img width="425" alt="Screenshot 2024-02-13 at 1 45 38 PM" src="https://github.com/Prisca92/Project-4-final/assets/140542648/a447217e-a50f-477f-b6dd-cc6aa3c3e50c"> 
 

