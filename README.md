# Project-4-final

Project Description

Contributors:
Prisca Gossin
Mehrin Khan 



Overview:

Our objective is to create a Machine learning model for detecting heart failure by focusing on various causative attributes. The aim is to predict early signs of heart failure in order to identify individuals who might be at risk of developing heart problems. By analyzing relevant data, we can anticipate and catch potential issues early on based on certain variables or features. This approach not only improves health outcomes but also ensures that healthcare resources are used more efficiently.


                                  Process



1. Reading data 

This is simply reading the data, specifying the type of variables, as well as which variable is the target to predict.



1.Data Pre-processing

This step was used to make the original data ready for model building. This Data pre-processing step included data imputation, variable scaling, converting categorical variable to numerical. In general, it is a series of tasks to make original data consumable and understandable for machine-learning models. 

We processed the data by:
-Normalizing the data
-Renaming columns
-Transforming columns to be numeric
-using pd.get_dummies() to convert categorical data to numeric
-Converting orginal data which was in an excel sheet to a true csv file

Resulting data: 918 features


2.Model building and selection

Supervised learning:

The model building process finds the model that fits best for the training data set in terms of prediction accuracy. One of the most popular approaches to achieve this goal is to iterate over multiple related machine learning models to see which one is the best fit. For this project we used three regression model classes – BalancedRandomForest, DecisionTree and Linear Regression – are fitted, and the one with highest r-square is picked as the best fit.  Both BalancedRandomforest and the linear regression model provided an accuracy score of 90%. Ultimately BalancedRandomforest classifier was the model we choose as the best fit as it is the most reliable an accurate model due to its high accuracy, feature important and versatility. In the Medical sector when making predictions it is best to use the most reliable machine learning model for prediction purposes. This system could also suggest precautions to the user. To better evaluate the random forest we used confusion matrix  which is a table that shows how many instances of each class were correctly of incorrectly predicted.Helps identify source errors like false positives and false negatives which we did not encounter.

3.Model deployment 

The model deployment step makes the machine-learning model available to make predictions based on an individuals health .
To deploy the model with flask we needed to first save our best fit model by using the Joblid  method which is an alternative to model saving because it is able to operate with large arrays and I find it to be a simple model .We used the dump functionality to save the model to the model_jlib file.The we needed to load the model by providing the file object to the load option storing it in the m_jlib variable , which can later be used for predictions. We then called the predict method on m_jlib and passed it a 2d array with values of 5000. To integrate Flask with an ML model, you typically load the trained model in your Flask application. Then, you define routes and views to handle requests, preprocess input data, and pass it to the model for predictions.

5.Inference

After model deployment, the model is accessible from a web application framework with the purpose to be user friendly option.
Users can call and query this model in real time to build on-demand reports and visualizations.  In this example, flask was used  with a machine learning model because of it’s ability to return the models prediction as a response to individuals.


A Tableau dashboard was also built with optimal variables to display …………


Tableau Link: [
](https://public.tableau.com/app/profile/mehrin.khan/viz/HeartDisease_17074856563830/Sheet1#1)https://public.tableau.com/app/profile/mehrin.khan/viz/HeartDisease_17074856563830/Sheet1#1


Conclusion:

Things to consider or try the next time around:

Dive in deeper into feature importance to re evaluate the performance of the random forest model to identify interactions between feature.
With the data already being pretty balanced I don’t think it was necessary to use BalancedforestClassifier, random forest could’ve been the model tested instead.



