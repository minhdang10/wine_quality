# A. Linear Regression
#setdirectory

#loadlibraries
library(leaps)
library(car)
library(MASS)
library("ggpubr")

#read and interpret data
data = read.csv("winequality-full.csv")
summary(data)
str(data)

#create holdout data
set.seed(1)
x = 1:nrow(data)
size = nrow(data)*0.8
train.index = sample(x,size,replace = FALSE)
train = data[train.index,]
test = data[-train.index,]

#Initial LR model with all variables
qpred = lm(quality~fixed.acidity+volatile.acidity+citric.acid+residual.sugar+chlorides+free.sulfur.dioxide+total.sulfur.dioxide+density+pH+sulphates+alcohol, data=train)

#plot and vif tests
par(mfrow=c(2,2))
par(mar=c(1,1,1,1))
plot(qpred)
vif(qpred)

# B. Logistic Regression and Random Forest with Regression Tree
# Read in data

df = read.csv('winequality-full.csv')

# Get an idea of our data, use descripitive stats as well

str(df)
summary(df)

prop.table(summary(df$color))
# about 75% white, 25% red

# Important to point out the wines are rated 3-9, no wine got 0,
# no wine got a perfect score
# ------------------------------------------------------
# EDA

# Histogram of quality, get an idea of the distribution
# mean is roughly 5.87, median is 6 exactly, pretty centered as the two values
# do not differ too much
par(mfrow = c(1,1))

hist(df$quality, col = 'violet', xlab = 'Quality', ylab = 'Number of Times Rated'
     , main = 'Histogram of Wine Quality', border = 'darkmagenta')

par(mfrow = c(1,2))

plot(as.factor(df$quality), df$alcohol, main = 'Alcohol Content by Quality', col = 'darkmagenta'
     , border = 'black', ylab = 'Alcohol Content', xlab = 'Quality')

plot(as.factor(df$quality), df$residual.sugar, main = 'Residual Sugar by Quality'
     , border = 'red', ylab = 'Residual Sugar', xlab = 'Quality')

# ----------------------------------------------------------

# Set up for model building

set.seed(1)

train.index = sample(1:nrow(df), nrow(df)*.8)

train = df[train.index,]
test = df[-train.index,]

# Create validation set, 20% of training

set.seed(2)
val.index = sample(1:nrow(train), nrow(train) *.2)
val_set = train[val.index,]



# -------------------------------------------------------
# We will be dealing with regression trees, so re-index training, test, and val sets

train = df[train.index,]
test = df[-train.index,]
val_set = train[val.index,]

reg.model = randomForest(quality ~., data = train, importance = TRUE)
reg.model

plot(reg.model, main = 'Number of Trees vs. Error', col = 'darkmagenta')   
# use to determine optimal number of trees, 100

reg.model2 = randomForest(quality ~., data = train, ntree = 100, importance = TRUE)
reg.model2

# Compare both models on validation set
# Most important variables the same for both models

pred.val = predict(reg.model, val_set)
mse.val = mean((pred.val - val_set$quality)^2)
mse.val

# Model 1 is marginally better, use that on test set

pred.test = predict(reg.model, test)
mse.test = mean((pred.test - test$quality)^2)
mse.test

varImpPlot(reg.model, main = 'Variable Importance Plots for Optimal Model')

# Logistic Regression

df$quality = ifelse(df$quality >= 6, 'High', 'Low')
summary(as.factor(df$quality))

set.seed(1)

train.index = sample(1:nrow(df),nrow(df)*0.80)

train = df[train.index,]
test = df[-train.index,]

model.log1 = glm(as.factor(quality)~., data = train, family = binomial)
summary(model.log1) # AIC = 5414.5

# Create model with only significant predictors


# Build model using the most significant predictors from model 2

model.log3 = glm(as.factor(quality) ~ volatile.acidity + 
                   residual.sugar + free.sulfur.dioxide + total.sulfur.dioxide
                 + sulphates + alcohol, data = train, family = binomial)

summary(model.log3)

exp(coef(model.log3))  # Analyze relationship

# Try kfoldcv
k = 5
fold = sample(1:k, nrow(train), replace = TRUE)

# Space to store the metrics from each iteration

kfold.acc = 1:k
kfold.sens = 1:k
kfold.prec = 1:k

# For loop to keep creating models and testing

for(i in 1:k) {
  test.kfold = train[fold==i,]
  train.kfold = train[fold!=i,]
  
  model.log3 = glm(as.factor(quality) ~ volatile.acidity + 
                     residual.sugar + free.sulfur.dioxide + total.sulfur.dioxide
                   + sulphates + alcohol, data = train, family = binomial)
  
  
  pred.log.val = predict(model.log3, test.kfold, type = 'response')
  pred.class.val = pred.log.val
  
  # Keep changing threshold based on ROC Curve
  pred.class.val[pred.log.val > 0.3] = "Low"
  pred.class.val[!pred.log.val > 0.3] = "High"
  
  c.matrix.log.val = table(actual = test.kfold$quality, pred.class.val)
  
  acc.log.val = (c.matrix.log.val[1] + c.matrix.log.val[4]) / (sum(c.matrix.log.val))
  sens.log.high.val = c.matrix.log.val[4] / (c.matrix.log.val[2] + c.matrix.log.val[4])
  prec.log.high.val = c.matrix.log.val[4] / (c.matrix.log.val[3] + c.matrix.log.val[4])
  
  kfold.acc[i] = acc.log.val
  kfold.sens[i] = sens.log.high.val
  kfold.prec[i] = prec.log.high.val
  
  
}

kfold.acc
kfold.sens
kfold.prec

mean(kfold.acc)
mean(kfold.sens)
mean(kfold.prec)
# Create model on training set using all predictors

model.log1 = glm(as.factor(quality)~., data = train, family = binomial)
summary(model.log1) # AIC = 5414.5

# Create model with only significant predictors


# Build model using the most significant predictors from model 2

model.log3 = glm(as.factor(quality) ~ volatile.acidity + 
                   residual.sugar + free.sulfur.dioxide + total.sulfur.dioxide
                 + sulphates + alcohol, data = train, family = binomial)

summary(model.log3)



###### USE EXCLUSIVELY FOR DETERMINING THRESHOLDS OF PROBABILITY

library(ROCR)
ROCRpred = prediction(pred.log.val, test.kfold$quality)
ROCRperf = performance(ROCRpred, 'tpr', 'fpr')

par(bg = 'white')
plot(ROCRperf, colorize = TRUE, print.cutoffs.at = seq(0,1 ,by = .1), text.adj = c(-.2, 1.7)
     , main = "ROC Curve")

# Evaluate now on test data ###########

pred.log.test = predict(model.log3, test, type = 'response')

pred.class.test = pred.log.test


pred.class.test[pred.log.test > 0.3] = "Low"
pred.class.test[!pred.log.test > 0.3] = "High"
pred.class.test[1:10]

c.matrix.log.test = table(actual = test$quality, pred.class.test)

c.matrix.log.test

acc.log.test = (c.matrix.log.test[1] + c.matrix.log.test[4]) / (sum(c.matrix.log.test))
sens.log.high.test = c.matrix.log.test[4] / (c.matrix.log.test[2] + c.matrix.log.test[4])
prec.log.high.test = c.matrix.log.test[4] / (c.matrix.log.test[3] + c.matrix.log.test[4])

data.frame(acc.log.test, sens.log.high.test, prec.log.high.test)

# C. K-Nearest Neighbor
setwd("/Users/heronoop/Desktop/CIS 3920/Project")
library(class)

data = read.csv("winequality-full.csv")
summary(data)
str(data)

normalize = function(x) {
  return ((x-min(x)) / (max(x)-min(x)))
}

norm = lapply(data[1:11],normalize)
data$quality = ifelse(data$quality >= 6, "high", "low")
norm.data = cbind(data[12],data[13],norm)

norm.data$color = as.character(norm.data$color)

norm.data$color[norm.data$color=="w"] = 0
norm.data$color[norm.data$color=="r"] = 1

summary(norm.data) 
str(norm.data)

set.seed(1)
x = 1:nrow(norm.data)
size = nrow(norm.data)*0.8

train.index = sample(x,size,replace = FALSE)

train = norm.data[train.index,]
test = norm.data[-train.index,]

train.x = train[,2:13]
test.x = test[,2:13]
train.cl = train[,1]

# set the stage for 10 odd K's between 1-20
rep = seq(1,20,2) 
rep.acc = rep
rep.sens = rep
rep.prec = rep

# index for 5-fold cv
set.seed(1)
k=5
fold = sample(1:k,nrow(train.x),replace=TRUE)

iter = 1 # index for rep iteration
for (K in rep) {
  # space to store metrics from each iteration of 5-fold cv
  kfold.acc = 1:k
  kfold.sens = 1:k
  kfold.prec = 1:k
  
  for (i in 1:k) {
    #data for test and training sets
    test.kfold = train.x[fold==i,]
    train.kfold = train.x[fold!=i,]
    
    # class labels for test and training sets
    test.cl.actual = train.cl[fold==i]
    train.cl.actual = train.cl[fold!=i]
    
    # make predictions on class labels for test set
    pred.class = knn(train.kfold,test.kfold,train.cl.actual,k=K)
    
    # evaluate metrics for "yes"
    c.matrix = table(test.cl.actual,pred.class)
    acc = mean(pred.class==test.cl.actual)
    sens.high = c.matrix[1]/(c.matrix[1]+c.matrix[3])
    prec.high = c.matrix[1]/(c.matrix[1]+c.matrix[2])
    
    # store result for each k-fold iteration
    kfold.acc[i] = acc
    kfold.sens[i] = sens.high
    kfold.prec[i] = prec.high
  }
  
  # store average k-fold perfomance for each KNN model
  rep.acc[iter] = mean(kfold.acc)
  rep.sens[iter] = mean(kfold.sens)
  rep.prec[iter] = mean(kfold.prec)
  
  iter = iter+1
}
# plot the results for each KNN model.
par(mfrow=c(1,3))
metric = as.data.frame(cbind(rep.acc,rep.sens,rep.prec))
color = c("blue","red","gold")
title = c("Accuracy","Sensitivity","Precision")

for (p in 1:3) {
  plot(metric[,p],type="b",col=color[p],pch=20,
       ylab="",xlab="K",main=title[p],xaxt="n")
  axis(1,at=1:10,labels=rep,las=2)
}

results = as.data.frame(cbind(rep,rep.acc,rep.sens,rep.prec))
names(results) = c("K","accuracy","sensitivity","precision")
results

# Predictions
pred.class = knn(train.x,test.x,train.cl,k=17)
c.matrix = table(test$quality,pred.class)
c.matrix

acc = mean(pred.class==test$quality)
sens.high = c.matrix[1]/(c.matrix[1]+c.matrix[3])
prec.high = c.matrix[1]/(c.matrix[1]+c.matrix[2])
as.data.frame(cbind(acc,sens.high,prec.high))

# D. Feature Engineering with KNN (full model with flavor)
setwd("/Users/heronoop/Desktop/CIS 3920/Project")
library(class)

data = read.csv("winequality-full_flavors.csv")
summary(data)
str(data)

data = cbind(data[10],data[1:9],data[12],data[11],data[13])
str(data)
normalize = function(x) {
  return ((x-min(x)) / (max(x)-min(x)))
}

norm = lapply(data[2:11],normalize)
data$quality = ifelse(data$quality >= 6, "high", "low")
norm.data = cbind(data[1],data[12],data[13],norm)

levels(norm.data$flavor)
norm.data$color = as.character(norm.data$color)

norm.data$color[norm.data$color=="w"] = 0
norm.data$color[norm.data$color=="r"] = 1

norm.data$dry[norm.data$flavor=="Dry"] = 1
norm.data$dry[norm.data$flavor!="Dry"] = 0

norm.data$off_dry[norm.data$flavor=="Off Dry"] = 1
norm.data$off_dry[norm.data$flavor!="Off Dry"] = 0

norm.data$semi_sweet[norm.data$flavor=="Semi Sweet"] = 1
norm.data$semi_sweet[norm.data$flavor!="Semi Sweet"] = 0

norm.data$sweet[norm.data$flavor=="Sweeet"] = 1
norm.data$sweet[norm.data$flavor!="Sweeet"] = 0

norm.data$flavor = NULL

summary(norm.data) 
str(norm.data)

set.seed(1)
x = 1:nrow(norm.data)
size = nrow(norm.data)*0.8

train.index = sample(x,size,replace = FALSE)

train = norm.data[train.index,]
test = norm.data[-train.index,]

train.x = train[,2:16]
test.x = test[,2:16]
train.cl = train[,1]

# set the stage for 10 odd K's between 1-20
rep = seq(1,20,2) 
rep.acc = rep
rep.sens = rep
rep.prec = rep

# index for 5-fold cv
set.seed(1)
k=5
fold = sample(1:k,nrow(train.x),replace=TRUE)

iter = 1 # index for rep iteration
for (K in rep) {
  # space to store metrics from each iteration of 5-fold cv
  kfold.acc = 1:k
  kfold.sens = 1:k
  kfold.prec = 1:k
  
  for (i in 1:k) {
    #data for test and training sets
    test.kfold = train.x[fold==i,]
    train.kfold = train.x[fold!=i,]
    
    # class labels for test and training sets
    test.cl.actual = train.cl[fold==i]
    train.cl.actual = train.cl[fold!=i]
    
    # make predictions on class labels for test set
    pred.class = knn(train.kfold,test.kfold,train.cl.actual,k=K)
    
    # evaluate metrics for "yes"
    c.matrix = table(test.cl.actual,pred.class)
    acc = mean(pred.class==test.cl.actual)
    sens.high = c.matrix[1]/(c.matrix[1]+c.matrix[3])
    prec.high = c.matrix[1]/(c.matrix[1]+c.matrix[2])
    
    # store result for each k-fold iteration
    kfold.acc[i] = acc
    kfold.sens[i] = sens.high
    kfold.prec[i] = prec.high
  }
  
  # store average k-fold perfomance for each KNN model
  rep.acc[iter] = mean(kfold.acc)
  rep.sens[iter] = mean(kfold.sens)
  rep.prec[iter] = mean(kfold.prec)
  
  iter = iter+1
}
# plot the results for each KNN model.
par(mfrow=c(1,3))
metric = as.data.frame(cbind(rep.acc,rep.sens,rep.prec))
color = c("blue","red","gold")
title = c("Accuracy","Sensitivity","Precision")

for (p in 1:3) {
  plot(metric[,p],type="b",col=color[p],pch=20,
       ylab="",xlab="K",main=title[p],xaxt="n")
  axis(1,at=1:10,labels=rep,las=2)
}

results = as.data.frame(cbind(rep,rep.acc,rep.sens,rep.prec))
names(results) = c("K","accuracy","sensitivity","precision")
results

pred.class = knn(train.x,test.x,train.cl,k=19)
c.matrix = table(test$quality,pred.class)
c.matrix

# Predictions
acc = mean(pred.class==test$quality)
sens.high = c.matrix[1]/(c.matrix[1]+c.matrix[3])
prec.high = c.matrix[1]/(c.matrix[1]+c.matrix[2])
as.data.frame(cbind(acc,sens.high,prec.high))

# E. Feature Engineering with Classification Random Forest  
# with full model of flavor
setwd("/Users/heronoop/Desktop/CIS 3920/Project")
library(randomForest)
library(class)

data = read.csv("winequality-full_flavors.csv")
summary(data)
str(data)

data = cbind(data[10],data[1:9],data[12],data[11],data[13])
str(data)

set.seed(1)
x = 1:nrow(data)
size = nrow(data)*0.8

train.index = sample(x,size,replace = FALSE)

train = data[train.index,]
test = data[-train.index,]

train$quality = ifelse(train$quality>=6,"high","low")
test$quality = ifelse(test$quality>=6,"high","low")
table(as.factor(train$quality))

set.seed(1)
reg.model = randomForest(as.factor(quality)~.,data=train, importance = TRUE)
reg.model 

# 2.
pred.class = predict(reg.model,test,type = "class")
table(pred.class)

c.matrix = table(test$quality,pred.class);  c.matrix
acc = mean(test$quality==pred.class)
sens = c.matrix[1]/(c.matrix[1]+c.matrix[3])
prec = c.matrix[1]/(c.matrix[1]+c.matrix[2])
data.frame(acc,sens,prec)

# 3.
importance(reg.model)
varImpPlot(reg.model,main="Variable Importance Plot")
