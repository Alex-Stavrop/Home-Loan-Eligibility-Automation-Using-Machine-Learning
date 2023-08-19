#Libraries
library(readr)
library(ROSE)
library(caret)
library(caTools)
library(yardstick)
library(ggplot2)
library(gridExtra)
library(dplyr)
library(tidyverse)
library(xgboost)
library(neuralnet)
library(mlbench)
library(Ckmeans.1d.dp)
library(kableExtra)
library(formattable)
#Load Data Set
Data <- read_csv("Loan_Data.csv")
View(Data)
summary(Data)
#Check for missing values
sapply(X = Data, FUN = function(x) sum(is.na(x)))
df<-na.omit(Data)
sapply(X = df, FUN = function(x) sum(is.na(x)))
df<- df[,-1]
#Check balance of data set in regard with loan status
barplot(prop.table(table(df$Loan_Status)),
        col = rainbow(2),
        ylim = c(0, 1),
        main = "Class Distribution of Loan Status Before Oversampling")
#Oversample the data set
df1 <- ovun.sample(Loan_Status ~., data = df, method = "over",seed = 123)$data
barplot(prop.table(table(df1$Loan_Status)),
        col = rainbow(2),
        ylim = c(0, 1),
        main = "Class Distribution of Loan Status After Oversampling")
# Factorise variables
df1$Gender <- as.factor(df1$Gender)
df1$Married <- as.factor(df1$Married)
df1$Dependents <- as.factor(df1$Dependents)
df1$Education <- as.factor(df1$Education)
df1$Self_Employed <- as.factor(df1$Self_Employed)
df1$Credit_History <- as.factor(df1$Credit_History)
df1$Property_Area <- as.factor(df1$Property_Area)
df1$Loan_Status <- as.factor(df1$Loan_Status)
#Standarise numeric variables for knn model 
df1 <- df1 %>% mutate_at(c('ApplicantIncome',"CoapplicantIncome","LoanAmount",
                           "Loan_Amount_Term"), ~(scale(.) %>% as.vector))
#Split data into train and test set
set.seed(1234)
split <- sample.split(df1, SplitRatio = 0.7) 
data_train <- subset(df1, split == "TRUE") 
data_test <- subset(df1, split == "FALSE") 

#Develop Classification Models

#Fit a knn model 
set.seed(100)
trControl <- trainControl(method  = "cv",
                          number  = 10)
knn <- train(Loan_Status ~ .,
             method     = "knn",
             tuneGrid   = expand.grid(k = 1:25),
             trControl  = trControl,
             metric     = "Accuracy",
             data       = data_train)
plot(knn)
#make prediction on test set with knn model
knn_pred <- predict(knn, data_test)
#Confusion matrix
truth_predicted<-data.frame(
  obs = data_test$Loan_Status,
  pred = knn_pred)
cfm <- conf_mat(truth_predicted, obs, pred)
#Plot CM
autoplot(cfm, type = "heatmap") +
  scale_fill_gradient(low="#D6EAF8",high = "#2E86C1") +
  ggtitle("Confusion Matrix - Prediction with KNN model
          Accuracy: 81,73%") 


#Fit a random forest
set.seed(1000)
# Define the control and tune
trControl <- trainControl(method = 'cv',
                          number = 10,
                          search = "grid")
tuneGrid <- expand.grid(.mtry = c(1:10))
#model
rf_mtry <- train(Loan_Status ~.,
                 data = data_train,
                 method = "rf",
                 metric = "Accuracy",
                 tuneGrid = tuneGrid,
                 trControl = trControl,
                 importance = TRUE,
                 nodesize = 14,
                 ntree = 1000)
plot (rf_mtry)
#Make predictions on test set with Random Forest
rf_pred <- predict(rf_mtry, data_test)
#Confusion matrix
truth_predicted1<-data.frame(
  obs = data_test$Loan_Status,
  pred = rf_pred)
cfm1 <- conf_mat(truth_predicted1, obs, pred)
#Plot Confusion Matrix
autoplot(cfm1, type = "heatmap") +
  scale_fill_gradient(low="#D6EAF8",high = "#2E86C1") +
  ggtitle("Confusion Matrix - Prediction with Random Forest 
          Accuracy: 79,90%")

#Fit a XGBoost model 
set.seed(10000)
xgboost <- train(Loan_Status ~., 
                 data = data_train, 
                 method = "xgbTree",
                 metric= "Accuracy",
                 trControl = trainControl("cv",number = 10),
                 verbosity = 0)
xgboost$bestTune
plot(xgboost)
xgboost
#make prediction on test set with xgboost model
xgboost_pred <- predict(xgboost, data_test)
#Confusion matrix
truth_predicted2<-data.frame(
  obs = data_test$Loan_Status,
  pred = xgboost_pred)
cfm2 <- conf_mat(truth_predicted2, obs, pred)
#Plot confusion matrix
autoplot(cfm2, type = "heatmap") +
  scale_fill_gradient(low="#D6EAF8",high = "#2E86C1") +
  ggtitle("Confusion Matrix - Prediction with XGBoost model
          Accuracy: 83,56% ") 
#Create a Table with all models for comparison
Models = data.frame(matrix(c(0), nrow=3, ncol=9))
colnames(Models) = (c('Models','True Pos', 'True Neg','False Pos', 'False Neg'
                      ,'Overall Error Rate %', 'Accuracy %','Sensitivity %',
                      'Specificity %'))
Models$Models = (c('KNN', 'RandomForest', 'XGBoost'))

#Transfer results to the Model Comparison Table 
# KNN model
Models[1,2] <-cfm$table[2,2]
Models[1,3] <-cfm$table[1,1]
Models[1,4] <-cfm$table[2,1]
Models[1,5] <-cfm$table[1,2]
# Random Forest
Models[2,2] <-cfm1$table[2,2]
Models[2,3] <-cfm1$table[1,1]
Models[2,4] <-cfm1$table[2,1]
Models[2,5] <-cfm1$table[1,2]

# XGBoost
Models[3,2] <-cfm2$table[2,2]
Models[3,3] <-cfm2$table[1,1]
Models[3,4] <-cfm2$table[2,1]
Models[3,5] <-cfm2$table[1,2]

# Calculate the formula
Overall_Error_Rate = (Models[,4] + Models[,5])/
  (Models[,2] + Models[,3]+ Models[,4] + Models[,5])*100
Sensitivity = Models[,2]/(Models[,2] + Models[,5])*100 
Specificity = Models[,3]/(Models[,4] + Models[,3])*100

# Calculate the results
Models[1:3,6]=round(Overall_Error_Rate, digits = 3)
Models[1:3,8]=round(Sensitivity, digits = 3)
Models[1:3,9]=round(Specificity, digits = 3)


# Calculate the overall Accuracy again
Overall_Accuracy = round((100-Models[,6]), digits = 3)
Models[1:3,7]=Overall_Accuracy

# Format table with Kable extra
knitr::kable(Models,
             caption = "Models Comparison") %>%
  kable_styling(font_size = 10,full_width = T,latex_options = "hold_position")

#Features importance of best model
xgb_imp <- xgb.importance(feature_names = xgboost$finalModel$feature_names,
                          model = xgboost$finalModel)
xgb.ggplot.importance(xgb_imp)
