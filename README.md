# Project-4-final

Project Description

Contributors:
Prisca Gossin,
Mehrin Khan 

Built With
* Python and Packages (eg. scikit-learn, matplotlib)
* Tableau


Overview:

Our objective is to create a machine learning model for detecting heart disease by focusing on various causative attributes, aiming to predict early signs of heart disease to identify individuals at risk. Analyzing relevant data allows us to anticipate and catch potential issues early on based on specific variables or features, improving health outcomes and ensuring efficient use of healthcare resources. Health care professionals will provide recommendations for managing and reducing the risk of heart disease, including lifestyle modifications and preventive measures, to improve overall health and minimize disease risk for at-risk patients.
                                  
                                  Process



1. Data Collection/ Reading data

This is simply reading the data, specifying the type of variables, as well as which variable is the target to predict.

Data was collected from Kaggle: 



1.Data Pre-processing

This step was used to make the original data ready for model building. This Data pre-processing step included data imputation, variable scaling, converting categorical variable to numerical. In general, it is a series of tasks to make original data consumable and understandable for machine-learning models. 

We processed the data by:
-Normalizing the data
/Users/priscagossin/Desktop/Screenshot 2024-02-10 at 11.53.05 AM.png
-Renaming columns
-Transforming columns to be numeric
-using pd.get_dummies() to convert categorical data to numeric
-Converting orginal data which was in an excel sheet to a true csv file

Resulting data: 918 features


/Users/priscagossin/Desktop/Screenshot 2024-02-12 at 8.56.46 PM.png
/Users/priscagossin/Desktop/Screenshot 2024-02-12 at 8.57.29 PM.png


/Users/priscagossin/Desktop/Screenshot 2024-02-12 at 8.58.15 PM.png


/Users/priscagossin/Desktop/Screenshot 2024-02-12 at 8.56.46 PM.png

2.Model building and selection
/Users/priscagossin/Desktop/Screenshot 2024-02-12 at 10.03.16 PM.png
Supervised learning:

The model building process finds the model that fits best for the training data set in terms of prediction accuracy. One of the most popular approaches to achieve this goal is to iterate over multiple related machine learning models to see which one is the best fit. For this project we used three regression model classes – BalancedRandomForest, DecisionTree and Linear Regression – are fitted, and the one with highest r-square is picked as the best fit.  Both BalancedRandomforest and the linear regression model provided an accuracy score of 90%. Ultimately BalancedRandomforest classifier was the model we choose as the best fit as it is the most reliable an accurate model due to its high accuracy, feature important and versatility. In the Medical sector when making predictions it is best to use the most reliable machine learning model for prediction purposes. This system could also suggest precautions to the user. To better evaluate the random forest we used confusion matrix  which is a table that shows how many instances of each class were correctly of incorrectly predicted.Helps identify source errors like false positives and false negatives which we did not encounter.


Tableau Link: 


Conclusion:

Evaluation
    Utilized accuracy scores, confusion matrix and classification reports to
    compare to access the performance of all 3 models. We ultimately selected model 2 as the final model due to its overall superior performance. And used this model to create a Web application.

Things to consider or try the next time around:

Dive in deeper into feature importance to re evaluate the performance of the random forest model to identify interactions between features.
With the data already being pretty balanced I don’t think it was necessary to use BalancedforestClassifier, random forest could’ve been the model tested instead.


  
    
