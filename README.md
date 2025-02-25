
# Credit-Card-Default-Prediction


![This is an image](https://www.loansettlement.com/blog/wp-content/uploads/2022/01/Ls-B-22-Jan.jpg)

Developed a model to predict if the customer is going to default on the credit card payment or not. 




## Problem Statement

This is our ML Supervised Classfication Project.We will be looking at data set of credit card clients. We will predict if the client will default or not.The dataset contains information about default payments,credit data, history of payments, bill statements of credit card of the client in Taiwan.
## Project Description

This is our Classification 
Project,  we will be looking into
multiple classification models and try to
come up with a best model at the end of
this project. We are only focussing on all
that algorithm which has been taught to
us till now in our class. SVM, KNN,
Gradient Naive Bayes and a few more
algorithm we have implemented in this
capstone project.

ML pipeline to be followed

● Dataset Inspection
● Data Cleaning
● Exploratory Data Analysis
● Feature Engineering
● Handling Class Imbalance
● One Hot Encoding
● Baseline Model with default
parameters
● Performance Metrics
● Optimization of the Model
● Feature Importance
● Hyperparameter Tuning
● Analyze Results

![This is an image](https://media.giphy.com/media/C8RmYDE0F7020ja4f2/giphy.gif)

Challenges-

Extracting new features from
existing features were a bit
tedious job to do.

Handling Imbalance data in
Target features

There were few undefined data
records present in the dataset
and few duplicate records

The Hyperparametric tuning
using GridSearch cv was really
time consuming task and
required lots of patience until it
get executed

Selecting different parameter for
hyperparameter tuning and
finding the best parameters has
to follow a trial n error
technique,which is again a
challenging job.
## Demo



![This is an image](https://repository-images.githubusercontent.com/472374004/2a6e2471-df20-4ea0-8d77-58dbc7dcd52d)
## About the data

Data Fields

● ID: ID of each client

● LIMIT_BAL: Amount of the given
credit (NT dollar): it includes both
the individual consumer credit
and his/her family
(supplementary) credit.

● SEX: Gender(1=male,2=female)

● EDUCATION: (1=graduate
school, 2=university, 3=high
school, 4=others,
5=unknown,6=unknown)

● MARRIAGE:Marrital status
(1=married,2=single,3=others)

● AGE:Age (year)

● PAY_0,PAY_2,PAY_3,PAY_4,PAY
_5: History of past payment. We
tracked the past monthly
payment records (from April to
September, 2005) as follows: X6
= the repayment status in
September, 2005; X7 = the
repayment status in August,
2005; . . .;X11 = the repayment
status in April, 2005. The
measurement scale for the
repayment status is: -1 = pay
duly; 1 = payment delay for one
month; 2 = payment delay for two
months; . . .; 8 = payment delay
for eight months; 9 = payment
delay for nine months and above.

● BILL_AMT1,BILL_AMT2,BILL_A
MT3,BILL_AMT4,BILL_AMT5,BIL
L_AMT6: Amount of bill
statement (NT dollar). X12 =
amount of bill statement in
September, 2005; X13 = amount
of bill statement in August, 2005;

. . .; X17 = amount of bill
statement in April, 2005.

● PAY_AMT1,PAY_AMT2,PAY_AM
T3,PAY_AMT4,PAY_AMT5,PAY_
AMT6:Amount of previous
payment (NT dollar). X18 =
amount paid in September, 2005;
X19 = amount paid in August,
2005; . . .;X23 = amount paid in
April, 2005.

● default.payment.next.month:
Default payment(1=Yes, 0= No)
## Steps Involved


Null values Treatment
The data set had null values out of which we replaced some with the mean of the feature some by zero and dropped some observations which were almost filled with null values.



Exploratory Data Analysis
We performed univariate and bivariate analyses. This process helped us figure out various aspects and relationships among variables. It gave us a better idea of which feature behaves in which manner.

Encoding of categorical columns
We used One Hot Encoding(converting to dummy variables) to produce binary integers of 0 and 1 to encode our categorical features because categorical features that are in string format cannot be understood by the machine and needs to be converted to the numerical format.

Feature Engineering

We have derived few new features from bill
amount and pay amount columns.Few new features derived are-
Total bill amount feature ,total paid amount and pending payment amount.

We have also implemented binning on
one feature i.e AGE column inorder to
better train our model efficiently .Then
we have encoded the column and given
a new name to the feature as
AGE_Encoded.

Feature Importance

Feature Importance is the process of
assigning scores to each feature,
depending on how useful it is in
predicting the target variable
If we remove those features which are
least important and keep the most
important ones, this might allow us to
better predict our target variable.

Fitting different models
For modeling, we tried various algorithms like:


Logistic Regression
SVM
Decision Tree
Random Forest Classification
XGBoost Classification


Tuning the hyperparameters for better recall
It is necessary to tune the hyperparameters of respective algorithms to improve accuracy and avoid overfitting when using tree-based models such as Random Forest Classifier and XGBoost classifier. The best set of hyperparameters was determined using a grid search algorithm.







## Conclusion

1. The maximum number of credit card
holders in Taiwan were females and the
average credit card limit provided by the
credit card company to their respective
customers was 167484.32(NT Dollars).

2. The most number of credit card
holders were having university degree
education and the most of the
customers marriage status was Single,
who carries a credit card in Taiwan.

3. The highest proportion of credit card
holders were youth in the age of 29,thus
we can conclude that mostly credit
cards were popular among youths of
taiwan than the older people.

4. The Correlation between features and
target variable tells us the level of
education and financial stability of the
customers had high impact on the
default rate.

5. The data also conveys us that the
best indicator of delinquency is the
behavior of the customer, which has
been predominately seen in the past
couple of months payment repayment status .The heat map shows us the high correlation of payment repayment status with the target variable.

6. The females of age group 20-30 have
very high tendency to default payment
compared to males in all age brackets.

7. Compartively after hyperparameter
tuning the XGBoost Model comes out to
be the best model in terms of its
AUC_ROC score(0.875) and Recall
score(0.82) and we can predict with
87.45% accuracy, whether a customer is
likely to default next month.
The reason is, XGBoost has high
predictive power and is almost 10 times
faster than the other gradient boosting
techniques. It also includes a variety of
regularization which reduces overfitting
and improves overall performance.

8. The Second best model was the Support Vector Machine with a
AUC_ROC score of 0.875 and a Recall
score of 0.805 and we can predict with
87.5% accuracy, whether a customer is
likely to default next month.
SVM model perform well when we
select the proper kernel and the risk of
overfitting is compartively less. We have
used rbf kernel and after fine tunning the
model we got Cost C parameter best
value as 100 and gamma as 0.01.

9. But it would be worth using Logistic
Regression model for production since
we do not just need a reliable model
with good ROC_AUC Score but also a
model that is **quick and less complex.

10. Except Naive Bayes model,all the
models have got really good ROC_AUC
scores with a probability of 0.85 on an
average.

11. The Random Forest and KNN
models were really overfitting with
default parameters and we handle the
overfit in both these model by fine tuning
the model.

12. Demographics: we see that being
Female, More educated, Single and
between 30-40years old means a
customer is more likely to make
payments on time.
## References

https://www.stepchange.org/de




https://www.geeksforgeeks.org/
splitting-data-for-machine-lear
ning-models/
https://analyticsindiamag.com/

why-data-scaling-is-important-
in-machine-learning-how-to-eff
ectively-do-it/

https://www.ibm.com/cloud/learn/random-forest

https://www.geeksforgeeks.org/
xgboost/

https://medium.com/analytics-vidhya/what-is-balance-and-imb
alance-dataset-89e8d7f46bc5

https://www.geeksforgeeks.org/

ml-handling-imbalanced-data-
with-smote-and-near-miss-algo

rithm-in-python/
https://www.educative.io/blog/o
ne-hot-encoding

