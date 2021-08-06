# Logistic-Regression-R--Tutorial

Logistic Regression R, In this tutorial we used the student application dataset for logistic regression analysis.

Logistic regression is a statistical model that in its basic form uses a logistic function to model a binary dependent variable.

In this tutorial, the target variable or dependent variable is Admit (0-No, 1-Yes) and the remaining variables are predictors or independent variables like GRE, GPA, and Rank.

The objective is to classify student applications as admit or reject.

Let’s read the student dataset in R.


Getting Data

mydata <- read.csv("D:/RStudio/LogisticRegression/binary.csv", header = T)
str(mydata)
'data.frame': 400 obs. of  4 variables:
 $ admit: int  0 1 1 1 0 1 1 0 1 0 ...
 $ gre  : int  380 660 800 640 520 760 560 400 540 700 ...
 $ gpa  : num  3.61 3.67 4 3.19 2.93 3 2.98 3.08 3.39 3.92 ...
 $ rank : int  3 3 1 4 4 2 1 2 3 2 ...

You can see a total of 400 observations and 4 variables in the dataset. Admit and Rank variables should be factor variables just convert those variables based on as. Factor.

mydata$admit <- as.factor(mydata$admit)
mydata$rank <- as.factor(mydata$rank)
Two-way table of factor variables

Let’s do a cross-validation before doing further analysis, for that we need to create xtabs and the idea should be there is no values in the table.

xtabs(~admit + rank, data = mydata)
rank
admit  1  2  3  4
    0 28 97 93 55
    1 33 54 28 12

The above dataset doesn’t have any zero values, we can proceed for further analysis


Data Partition

As usual create training and test datasets basis 80:20 ratio.

set.seed(1234)
ind <- sample(2, nrow(mydata), replace = T, prob = c(0.8, 0.2))
train <- mydata[ind==1,]
test <- mydata[ind==2,]



mymodel <- glm(admit ~ gpa+gre + rank, data = train, family = 'binomial')
summary(mymodel)

glm indicates generalized linear model and family is binomial because admit variable has only o and 1. Lets look at the summary of the model.

Call:
glm(formula = admit ~ gpa + gre + rank, family = "binomial",
    data = train)
Deviance Residuals:
   Min      1Q  Median      3Q     Max 
-1.587  -0.868  -0.618   1.130   2.118 
Coefficients:
            Estimate Std. Error z value Pr(>|z|)   
(Intercept) -5.00951    1.31651   -3.81  0.00014 ***
gpa          1.16641    0.38890    3.00  0.00271 **
gre          0.00163    0.00122    1.34  0.18018   
rank2       -0.57098    0.35827   -1.59  0.11101   
rank3       -1.12534    0.38337   -2.94  0.00333 **
rank4       -1.53294    0.47738   -3.21  0.00132 **
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
(Dispersion parameter for binomial family taken to be 1)
    Null deviance: 404.39  on 324  degrees of freedom
Residual deviance: 369.99  on 319  degrees of freedom
AIC: 382
Number of Fisher Scoring iterations: 4

More stars indicate more statistical significance. In this case gre and level rank 2 is not statistically significant. Let’s drop gre and re-run the model because gre is not significant.



mymodel <- glm(admit ~ gpa + rank, data = train, family = 'binomial') 
summary(mymodel) 

The new model AIC is better than the previous model, let’s check the same.

Call:
glm(formula = admit ~ gpa + rank, family = "binomial", data = train)
Deviance Residuals:
   Min      1Q  Median      3Q     Max 
-1.516  -0.888  -0.632   1.109   2.169 
Coefficients:
            Estimate Std. Error z value
(Intercept)   -4.727      1.292   -3.66
gpa            1.374      0.359    3.83
rank2         -0.571      0.356   -1.60
rank3         -1.165      0.380   -3.06
rank4         -1.564      0.476   -3.29
            Pr(>|z|)   
(Intercept)  0.00025 ***
gpa          0.00013 ***
rank2        0.10898   
rank3        0.00220 **
rank4        0.00101 **
---
Signif. codes: 
  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’
  0.1 ‘ ’ 1
(Dispersion parameter for binomial family taken to be 1)
    Null deviance: 404.39  on 324  degrees of freedom
Residual deviance: 371.81  on 320  degrees of freedom
AIC: 381.8
Number of Fisher Scoring iterations: 4

Now you can see that gpa pvalue goes down further compared to the previous model.
Prediction

Let’s do the prediction based on the above model

p1 <- predict(mymodel, train, type = 'response')
head(p1)
  1    2    3    4    6    7
0.28 0.30 0.68 0.13 0.24 0.35

Now you can see only 28% of chances are there the first candidate to admit the project. Similarly second candidate 29%, third candidate 68% and so on..

Regression Analysis Multicollinearity Problem Handling

head(train)
admit gre gpa rank
1     0 380 3.6    3
2     1 660 3.7    3
3     1 800 4.0    1
4     1 640 3.2    4
6     1 760 3.0    2
7     1 560 3.0    1

Now the predicted values are in terms of probability, we need to convert the predicted values into 0 and 1 because our train dataset contains dependent variables in the form 0 and 1.

Misclassification error – train data

pred1 <- ifelse(p1>0.5, 1, 0)
tab1 <- table(Predicted = pred1, Actual = train$admit)
tab1
Actual
Predicted   0   1
        0 208  73
        1  15  29

Based on 208+29 = 237 correct classifications and 73+15 =88 misclassifications. Let’s calculate the misclassifications error rate based on below code

1 - sum(diag(tab1))/sum(tab1)

misclassifications error rate is 27%.


Misclassification error – test data

p2 <- predict(mymodel, test, type = 'response')
pred2 <- ifelse(p2>0.5, 1, 0)
tab2 <- table(Predicted = pred2, Actual = test$admit)
tab2
Actual
Predicted  0  1
        0 48 20
        1 2  5

Total 48+5=53 correct classification and 21 misclassifications in test data.

1 - sum(diag(tab2))/sum(tab2)

misclassifications error rate is 29%.
Goodness-of-fit test

Logistic regression we can also try for the goodness of fittest.

with(mymodel, pchisq(null.deviance – deviance, df.null-df.residual, lower.tail = F))

the p-value is 0.00000145, p-value is too model and hence the model is statistically significant.
