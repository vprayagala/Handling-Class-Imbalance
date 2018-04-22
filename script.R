#Credit Card Fraud Detection 
###########################################################################
#Below are the methods used to treat imbalanced datasets:
#
#1.Undersampling
#2.Oversampling
#3.Synthetic Data Generation
#4.Cost Sensitive Learning

#We will be working with Synthetic Data Generation technique in this script
#ROSE (Random OverSampling Example) and SMOTE from DMwR
###########################################################################
#Import the Required Libraries
library(data.table)
library(C50)
library(DMwR)
library(caret)
library(ROCR)
library(pROC)
library(ROSE)
#Checking the working directories
rm(list=ls(all=T))
getwd()
setwd("/Users/Saatwik/Documents/Kaggle/CreditCardFraud/")
getwd()
#Read Data
data<-fread("creditcard.csv", stringsAsFactors = F, sep = ",", header =T)
#Random check of data
#Time and Amount are variables, V1 to V28 Principal components from PCA
dim(data)
str(data)
sum(is.na(data))

#Class distribution
prop.table(table(data$Class))

#Split the data into into train and test, check the class distribution
set.seed(7)
rows<-sample(1:nrow(data),0.7*nrow(data))
train <- data[rows,]
test <- data[-rows,]
#Check Class Distribution in each split
prop.table(table(train$Class))
prop.table(table(test$Class))
#change target variable into factor for classification.
train$Class <- as.factor(train$Class)
test$Class <- as.factor(test$Class)

####################################################################
#Before using smote, check with given data
model<-C5.0(Class~.,data=train)
predicted<-predict(model,test)
caret::confusionMatrix(test$Class, predicted)
roc.curve(test$Class, predicted, plotit = T,col="blue")
####################################################################
#DMwR package and use Synthetic Minority Oversampling Technique
#(SMOTE)
set.seed(7)
smot_train<- SMOTE(Class~., data = train, perc.over = 900, k = 5, perc.under = 850)
smot_test <- SMOTE(Class~., data = test, perc.over = 900, k = 5, perc.under = 850)
#datatable(smot_test)
prop.table(table(smot_train$Class))
prop.table(table(smot_test$Class))
smot_model<-C5.0(Class~.,data=smot_train)
p <- predict(smot_model, smot_test)
#Accuracy, Precision and Recall.
caret::confusionMatrix(smot_test$Class, p)

#ROC Curve
c <- c()
f <- c()
j <- 1

for(i in seq(0.01, 0.8 , 0.01)){
    set.seed(7)
    fit <- C5.0(Class~., data = smot_train)
    pre <- predict(fit, smot_test,type="prob")[,2]
    pre <- as.numeric(pre > i)
    auc <- roc(smot_test$Class, pre)
    c[j] <- i
    f[j] <- as.numeric(auc$auc)
    j <- j + 1
}
df <- data.frame(c = c, f = f)
p <- df$c[which.max(df$f)]
p
#model - randomforst build on train, use the same for above cutoff (p) value found
pre <- predict(smot_model, smot_test,type="prob")[,2]
#perf<-performance(prediction(abs(pre),smot_test$Class),"tpr","fpr")
#plot(perf,col="red")
roc.curve(smot_test$Class, pre, plotit = T,col="red",add.roc=T)
##Model performance
caret::confusionMatrix(smot_test$Class, as.factor(as.numeric(pre>p)))

###########################################################################
#Using ROSE package to deal with imbalance
rose_train <- ROSE(Class ~ ., data = train, seed = 1)$data
table(rose_train$Class)
rose_test <- ROSE(Class ~ ., data = test, seed = 1)$data
table(rose_test$Class)

rose_model<-C5.0(Class~.,data=rose_train)
p <- predict(rose_model, rose_test)
#Accuracy, Precision and Recall.
caret::confusionMatrix(rose_test$Class, p)
roc.curve(rose_test$Class, p, plotit = T,col="green",add.roc = T)
