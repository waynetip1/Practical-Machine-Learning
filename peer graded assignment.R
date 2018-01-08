library(caret)
library(dplyr)
train <- read.csv('pml-training.csv', na.strings = c("NA","#DIV/0!",""))
test <- read.csv('pml-testing.csv', na.strings = c("NA","#DIV/0!",""))
head(train)
str(train)
summary(train)
 
# check for na
any(is.na(train))

# keep columns that have 90% of the data, dropped 100 columns
train.noNA <- train[colSums(!is.na(train))>(nrow(train)*.9)]
test <- test[colSums(!is.na(test))>(nrow(test)*.9)]
any(is.na(train.noNA))

# Remove the first 7 columns because they are not needed for calculation
train.clean <- train.noNA[,-c(1:7)]

# get only numeric columns
num.cols <-sapply(train.clean,is.numeric)
# make correlation matrix
cor.data <- cor(train.clean[,num.cols])

# visualize correlation data
library(corrgram)
library(corrplot)
corrplot(cor.data, method = 'color')

# identify  correlated predictors for removal
high.cor <- findCorrelation(cor.data, cutoff = .75)
train.cor <- train.clean[,-high.cor]
train.cor$classe <-as.factor(train.cor$classe)

# split train.cor into training and testing sets
inTrain <- createDataPartition(y=train.cor$classe,p=0.75,list=FALSE)
training <- train.cor[inTrain,]
testing <- train.cor[-inTrain,]

# random forest
library(randomForest)
modelfit.rf <- randomForest(classe ~.,training)

# modelfit.rf <- train(classe~.,data=training,method='rf',
                     trControl=trainControl(method = 'cv',number=5),
                     prox=TRUE,allowParallel=TRUE)

# predict
pred.rf <- predict(modelfit.rf,testing)
# testing$classe <- pred.rf
cmatrix <- confusionMatrix(pred.rf,testing$classe)
cmatrix

# plot model fit
plot(modelfit.rf)
