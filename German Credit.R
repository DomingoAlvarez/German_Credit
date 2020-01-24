library(knitr)
library(tidyverse)
library(reshape2)
library(RColorBrewer)
library(GGally)
library(caret)
library(glmnet)
library(boot)
library(verification)

rm(list=ls())

german_credit <- read.table("http://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data")

colnames(german_credit) <- c("chk_acct", "duration", "credit_his", "purpose", 
                             "amount", "saving_acct", "present_emp", "installment_rate", "sex", "other_debtor", 
                             "present_resid", "property", "age", "other_install", "housing", "n_credits", 
                             "job", "n_people", "telephone", "foreign", "response")

german_credit$response <- german_credit$response - 1
german_credit$response <- as.factor(german_credit$response)

glimpse(german_credit)

summary(german_credit)

#Exploratory Data Analisys

#Duration
amount.mean <- german_credit %>% dplyr::select(amount, response) %>% group_by(response) %>% summarise(m =mean(amount))
duration.mean <- german_credit %>% dplyr::select(duration, response) %>%group_by(response) %>% summarise( m =mean(duration))



test.m <- german_credit[,c(2,5,8,13,16,18,21)]
test.m$response <- as.numeric(test.m$response)
ggplot(melt(german_credit[,c(2,21)]), aes(x = variable, y = value, fill = response)) + geom_boxplot() + xlab("response") + ylab("duration")


#Installment Rate

ggplot(german_credit, aes(factor(installment_rate), ..count..)) + 
  geom_bar(aes(fill = response), position = "dodge") + xlab("Installment Rates")


#Amount


ggplot(melt(german_credit[,c(5,21)]), aes(x = variable, y = value, fill = response)) + 
  geom_boxplot() + xlab("response") + ylab("amount")

#Age
ggplot(melt(german_credit[,c(13,21)]), aes(x = variable, y = value, fill = response)) + 
  geom_boxplot()+ xlab("response") + ylab("age")

#n_credits
ggplot(melt(german_credit[,c(16,21)]), aes(x = variable, y = value, fill = response)) + 
  geom_boxplot()

#chk

ggplot(german_credit, aes(chk_acct, ..count..)) + 
  geom_bar(aes(fill = response), position = "dodge") 

#credit_his
ggplot(german_credit, aes(credit_his, ..count..)) + 
  geom_bar(aes(fill = response), position = "dodge") 
#purpose
ggplot(german_credit, aes(purpose, ..count..)) + 
  geom_bar(aes(fill = response), position = "dodge") 
#saving_acct
ggplot(german_credit, aes(saving_acct, ..count..)) + 
  geom_bar(aes(fill = response), position = "dodge")
#other_debtor
ggplot(german_credit, aes(other_debtor, ..count..)) + 
  geom_bar(aes(fill = response), position = "dodge")
#sex
ggplot(german_credit, aes(sex, ..count..)) + 
  geom_bar(aes(fill = response), position = "dodge") 
#other_install
ggplot(german_credit, aes(other_install, ..count..)) + 
  geom_bar(aes(fill = response), position = "dodge") 
#foreign
ggplot(german_credit, aes(foreign, ..count..)) + 
  geom_bar(aes(fill = response), position = "dodge")

#Model

set.seed(1)
in.train <- createDataPartition(as.factor(german_credit$response), p=0.7, list=FALSE)
german_credit.train <- german_credit[in.train,]
german_credit.test <- german_credit[-in.train,]



#lasso
factor_var <- c(1,3,4,6,7,9,10,12,14,15,17,19,20,21)
num_var <- c(2,5,8,11,13,16,18)
train2 <- german_credit.train
train2[num_var] <- scale(train2[num_var])
train2[factor_var] <- sapply(train2[factor_var] , as.numeric)

X.train <- as.matrix(train2[,1:20])
Y.train <- as.matrix(train2[,21])

lasso.fit<- glmnet(x=X.train, y=Y.train, family = "binomial", alpha = 1)
plot(lasso.fit, xvar = "lambda", label=TRUE)

cv.lasso<- cv.glmnet(x=X.train, y=Y.train, family = "binomial", alpha = 1, nfolds = 10)
plot(cv.lasso)

cv.lasso$lambda.min
cv.lasso$lambda.1se

coef(lasso.fit, s=cv.lasso$lambda.min)
coef(lasso.fit, s=cv.lasso$lambda.1se)

credit.glm.final <- glm(response ~ chk_acct + duration + credit_his + saving_acct + present_emp + property, family = binomial, german_credit.train)

summary(credit.glm.final)

prob.glm1.insample <- predict(credit.glm.final, type = "response")
predicted.glm1.insample <- prob.glm1.insample > 0.2
predicted.glm1.insample <- as.numeric(predicted.glm1.insample)
mean(ifelse(german_credit.train$response != predicted.glm1.insample, 1, 0))

table(german_credit.train$response, predicted.glm1.insample, dnn = c("Truth", "Predicted"))


roc.plot(german_credit.train$response == "1", prob.glm1.insample)
roc.plot(german_credit.train$response == "1", prob.glm1.insample)$roc.vol$Area



#OOS misclassification rate and auc score

prob.glm1.outsample <- predict(credit.glm.final, german_credit.test, type = "response")
predicted.glm1.outsample <- prob.glm1.outsample > 0.2
predicted.glm1.outsample <- as.numeric(predicted.glm1.outsample)
table(german_credit.test$response, predicted.glm1.outsample, dnn = c("Truth", "Predicted"))


mean(ifelse(german_credit.test$response != predicted.glm1.outsample, 1, 0))

roc.plot(german_credit.test$response == "1", prob.glm1.outsample)
roc.plot(german_credit.test$response == "1", prob.glm1.outsample)$roc.vol$Area

#we use a 5:1 penalty for misclassification 

cost1 <- function(r, pi) {
  weight1 = 5
  weight0 = 1
  c1 = (r == 1) & (pi < 0.2)  
  c0 = (r == 0) & (pi > 0.2) 
  return(mean(weight1 * c1 + weight0 * c0))
}


cost1(german_credit.test$response,predicted.glm1.outsample)

#XGBoost and Variable Importance
#https://xgboost.readthedocs.io/en/latest/R-package/discoverYourData.html

require(xgboost)
require(Matrix)
require(data.table)

sparse_matrix <- sparse.model.matrix(response~.-1, data = german_credit.train)
head(sparse_matrix)

sparse_matrix_test<- sparse.model.matrix(response~.-1, data = german_credit.test)

output_vector = german_credit.train[,"response"] == 1

dtrain <- xgb.DMatrix(data = sparse_matrix, label = output_vector)

bst <- xgboost(data = dtrain, max.depth = 6,
                nrounds = 10,objective = "binary:logistic", eval_metric="error")




importance <- xgb.importance(feature_names = sparse_matrix@Dimnames[[2]], model = bst)
head(importance)


importanceRaw <- xgb.importance(feature_names = sparse_matrix@Dimnames[[2]], model = bst, data = sparse_matrix, label = output_vector)

# Cleaning for better display
importanceClean <- importanceRaw[,`:=`(Cover=NULL, Frequency=NULL)]

head(importanceClean)

#Plotting variable importance 
xgb.plot.importance(importance_matrix = importanceRaw)


#Prediction

pred <- predict(bst, sparse_matrix_test)

print(head(pred))

prediction <- as.numeric(pred > 0.5)
print(head(prediction))

err <- mean(as.numeric(pred > 0.5) != german_credit.test$response)
print(paste("test-error=", err))

#Confussion Matriz and Cost Matrix

table(german_credit.test$response, prediction, dnn = c("Truth", "Predicted"))
#OOS misclassification rate and auc score

prob.glm1.outsample <- predict(credit.glm.final, german_credit.test, type = "response")
predicted.glm1.outsample <- prob.glm1.outsample > 0.2
predicted.glm1.outsample <- as.numeric(predicted.glm1.outsample)
table(german_credit.test$response, predicted.glm1.outsample, dnn = c("Truth", "Predicted"))


mean(ifelse(german_credit.test$response != predicted.glm1.outsample, 1, 0))

roc.plot(german_credit.test$response == "1", prob.glm1.outsample)
roc.plot(german_credit.test$response == "1", prob.glm1.outsample)$roc.vol$Area

#we use a 5:1 penalty for misclassification 

cost1 <- function(r, pi) {
  weight1 = 5
  weight0 = 1
  c1 = (r == 1) & (pi < 0.2)  
  c0 = (r == 0) & (pi > 0.2) 
  return(mean(weight1 * c1 + weight0 * c0))
}


cost1(german_credit.test$response,predicted.glm1.outsample)
