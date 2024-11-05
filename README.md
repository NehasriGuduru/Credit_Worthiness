# Credit_Worthiness
Creditworthiness Prediction Model
This project is designed to create a credit scoring model that can predict the creditworthiness of a person by using available historical financial data. With the use of a dataset consisting of multiple financial attributes such as income, loan amounts, payment history, and credit card usage, we classified the individuals into different classes of credit scores using classification algorithms: "Good," "Standard," and "Poor."

Data Preprocessing and Feature Engineering
The dataset was cleaned and preprocessed for dealing with missing values and categorical variables. Label encoding has been done to convert categorical features into their respective numeric representations, making training models feasible. StandardScaler is applied to standardize features such that the model can be converged faster as well as perform better.
Model Development
Since a Random Forest Classifier is robust in the presence of non-linear relations, it was selected. The split dataset was set up as 80 percent training and the other 20 percent testing to ensure thorough testing of the model.
 
Evaluation Metrics
A performance of the model was done using metrics such as accuracy, precision, recall, and F1-score. The value of all the four metrics was high, which could suggest a high accuracy in distinguishing different creditworthiness classes.

Data Visualization
By the use of Seaborn and Matplotlib, one managed to create thought-provoking visualizations that would potentially help better understand relationships among several features and credit scores. For instance, generating box plots was possible so as to describe the way credit scores are distributed with different income levels or number of delayed payments so that key patterns and areas where credit management could improve will be revealed.

This creditworthiness prediction model gives the financial institution the risk assessment and will allow it to make informed lending decisions for financial stability.
