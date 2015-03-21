---
title: "Practical Machine Learning course Project"
output: html_document
---  

####Summary
The goal of this project to come up with a prediction algorithm for classifying activity of the excercise enthusiasts who take measurements to improve their health into various categories. Depending on the quality and quantiy of the activities measured, they are classified into one of the categories A, B, C, D or E. The variable 'classe' in training dataset holds this value 
Once the prediction algorithm is run and chosen using training dataset, it is applied on the test dataset that has 20 records and results are submitted as a separate assignment. 

####Data Sources
The data for the doing this project comes from http://groupware.les.inf.puc-rio.br/har.   
There are two data sets used in this project. One for training and one for testing
The links for downloading the data set are given below

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

####Prepare Data
The data is read from the url mentioned above using methods from RCurl package and cleaned to remove NA's and other missing and incorrect values. 


```r
library(RCurl)
url1 <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
url2 <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

f1 <- getURL(url1, ssl.verifypeer = FALSE)
f2 <- getURL(url2, ssl.verifypeer = FALSE)

trainCSV <- read.csv(textConnection(f1), na.strings=c("NA","#DIV/0!",""))
testCSV <- read.csv(textConnection(f2), na.strings=c("NA","#DIV/0!",""))
```

####Preprocess Data
Training data set is split into two - one for training the model and one for cross validation. caret package has been used to do this. Once the data is partitioned, the variables that doesn't have any impact on classe are identified using nearZeroVar method. These variables are then excluded from analysis. The identification variables in columns 1 to 6 like id, name and timestamps are also removed since these variables also won't have any impact on performance.


```r
library(caret)
inTrain <- createDataPartition(y=trainCSV$classe, p=0.75, list=FALSE)
train <- trainCSV[inTrain, ]
test <- trainCSV[-inTrain, ]
nearZeroVars <- nearZeroVar(train, saveMetrics=TRUE)
train <- train[,rownames(nearZeroVars[which(nearZeroVars$nzv!=TRUE),])]
train <- train[, -c(1:6)] #get rid of identity and time
```
Once the unwanted columns are removed, the columns with too many NAs are removed. Otherwise the random forest classification method will fail. 

```r
colTodelete = vector()
for(i in 1:length(train)) { 
        if( sum( is.na( train[, i] ) ) /nrow(train) >= .7 )  
        {  
          colTodelete = c(colTodelete,-i)        
        }  
}
train <- train[,c(colTodelete)]
```
In test data set for validation, only the columns present in train dataset are retained and for prediction, the 'classe' variable is also removed. The test dataset the 'classe' variable is retained for creating the confusion matrix to find out the accuracy of the method.  

```r
requiredCols <- colnames(train)
test <- test[requiredCols]
test1 <- subset(test, select=-c(classe))
```
At this stage data preparation is complete. Now the classification is done using two methods -  rpart and Random Forest. The packages rpart and randomForest are used for this. Cross Validation is done by applying the model on the test partition and analysing the confusion matrix results.

####Classification and Cross Validation
Let's first use rpart method to train the model and see the accuracy and error rates

```r
library(rpart)
library(randomForest)
rpartFit <- rpart(classe ~ ., data=train, method="class")
predRpart <- predict(rpartFit, test1, type = "class")
confusionMatrix(predRpart, test$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1300  197   30   90   31
##          B   36  466   42   16   39
##          C   24   85  687  125  104
##          D   15   65   45  506   49
##          E   20  136   51   67  678
## 
## Overall Statistics
##                                           
##                Accuracy : 0.7416          
##                  95% CI : (0.7291, 0.7538)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.6713          
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9319  0.49104   0.8035   0.6294   0.7525
## Specificity            0.9008  0.96637   0.9165   0.9576   0.9316
## Pos Pred Value         0.7888  0.77796   0.6702   0.7441   0.7122
## Neg Pred Value         0.9708  0.88780   0.9567   0.9295   0.9436
## Prevalence             0.2845  0.19352   0.1743   0.1639   0.1837
## Detection Rate         0.2651  0.09502   0.1401   0.1032   0.1383
## Detection Prevalence   0.3361  0.12215   0.2090   0.1387   0.1941
## Balanced Accuracy      0.9164  0.72871   0.8600   0.7935   0.8420
```
As per the confusion matrix, the accuracy is 0.7051 or 70.51% and the out of sample error rate is 29.49%

Now let's look at the Random Forest results


```r
rfFit <- randomForest(classe ~. , data=train)
predRF <- predict(rfFit, test1, type = "class")
confusionMatrix(predRF, test$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1394    8    0    0    0
##          B    0  940   13    0    0
##          C    1    1  839    4    0
##          D    0    0    3  800    1
##          E    0    0    0    0  900
## 
## Overall Statistics
##                                          
##                Accuracy : 0.9937         
##                  95% CI : (0.991, 0.9957)
##     No Information Rate : 0.2845         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.992          
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9993   0.9905   0.9813   0.9950   0.9989
## Specificity            0.9977   0.9967   0.9985   0.9990   1.0000
## Pos Pred Value         0.9943   0.9864   0.9929   0.9950   1.0000
## Neg Pred Value         0.9997   0.9977   0.9961   0.9990   0.9998
## Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
## Detection Rate         0.2843   0.1917   0.1711   0.1631   0.1835
## Detection Prevalence   0.2859   0.1943   0.1723   0.1639   0.1835
## Balanced Accuracy      0.9985   0.9936   0.9899   0.9970   0.9994
```
Here the accuracy is 0.9957 and the out of sample error rate is 0.0043

Obviously Random Forest method has better prediction accuracy and out of sample error rate. So we use Random Forest predictor for predicting our test dataset  

####Prediction for Test dataset
Now we do the prediction of the test dataset. The results are submitted to coursera website.

```r
testDataset <- testCSV[colnames(test1)]
predTest <- predict(rfFit, testDataset, type = "class")
```
The results of prediction are as given below

```r
predTest
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```
