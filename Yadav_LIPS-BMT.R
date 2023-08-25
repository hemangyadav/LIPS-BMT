setwd("/LIPS-BMT")

library(tidyverse)
library(caret)
library(pROC)
library(patchwork)

studypop <- read_csv('Data.csv')
studypop <- studypop %>% filter (Total.Hosp.LOS >= 1) # Filter patients > 24h stay
studypop <- studypop %>% select (-c(Total.Hosp.LOS, ClinicNumber)) # Remove LOS data + Identifying data 
set.seed(10)
studypop$HospID <- factor(studypop$HospID)
studypop$ARDS.Final <- factor(studypop$ARDS.Final) # Recode categorical variables as factors 
studypop$ARDS.Final <- recode(studypop$ARDS.Final, '1' = 'Yes', '0' = 'No', levels = c("No","Yes")) # Assign factors as labels + set order. This can be done as a single step with the prior command. 
studypop$ARF.4L.O2 <- factor(studypop$ARF.4L.O2)
studypop$ARF.4L.O2 <- recode(studypop$ARF.4L.O2, '1' = 'Yes', '0' = 'No', levels = c("No","Yes"))                      
studypop$Vent <- factor(studypop$Vent)
studypop$Vent <- recode(studypop$Vent, '1' = 'Yes', '0' = 'No', levels = c("No","Yes"))  
studypop$Tx.Type <- recode(studypop$Tx.Type,"Syngenic" = "Allogeneic") # For modeling purposes, syngenic transplants (rare) were grouped with allogeneic 
studypop$Tx.Type <- factor(studypop$Tx.Type, levels = c("Autologous", "Allogeneic"))
studypop <- studypop %>% mutate(Chemo = ifelse((Thalidomide == 1 | Cisplatin == 1 | Methotrexate == 1 | Carboplatin == 1), 1, 0))  
studypop$Chemo <- factor(studypop$Chemo)
studypop$Chemo <- recode(studypop$Chemo, '1' = 'Yes', '0' = 'No', levels = c("No","Yes"))
studypop$Smoking <- factor(studypop$Smoking)
studypop$Smoking <- recode(studypop$Smoking, '1' = 'Yes', '0' = 'No', levels = c("No","Yes"))  
studypop$DM <- factor(studypop$DM) 
studypop$DM <- recode(studypop$DM, '1' = 'Yes', '0' = 'No', levels = c("No","Yes"))  
studypop$Steroids.30days.PreAdmit <- factor(studypop$Steroids.30days.PreAdmit)
studypop$Steroids.30days.PreAdmit <- recode(studypop$Steroids.30days.PreAdmit, '1' = 'Yes', '0' = 'No', levels = c("No","Yes"))
studypop$Pneumonia.24h <- factor(studypop$Pneumonia.24h)
studypop$Pneumonia.24h <- recode(studypop$Pneumonia.24h, '1' = 'Yes', '0' = 'No', levels = c("No","Yes"))
studypop$Sepsis.24h <- factor(studypop$Sepsis.24h)
studypop$Sepsis.24h <- recode(studypop$Sepsis.24h, '1' = 'Yes', '0' = 'No', levels = c("No","Yes"))
studypop$Septic.Shock.24h <- factor(studypop$Septic.Shock.24h)
studypop$Septic.Shock.24h <- recode(studypop$Septic.Shock.24h, '1' = 'Yes', '0' = 'No', levels = c("No","Yes"))
studypop$CARV <- factor(studypop$CARV)
studypop$CARV <- recode(studypop$CARV, '1' = 'Yes', '0' = 'No', levels = c("No","Yes"))
studypop$Opioids <- factor(studypop$Opioids)
studypop$Opioids <- recode(studypop$Opioids, '1' = 'Yes', '0' = 'No',  levels = c("No","Yes"))
studypop$RBC.PLT.First24h <- factor(studypop$RBC.PLT.First24h)
studypop$RBC.PLT.First24h <- recode(studypop$RBC.PLT.First24h, '1' = 'Yes', '0' = 'No', levels = c("No","Yes"))
glimpse(studypop)

# Separate categorical and continuous data so as to impute missing continuous data variables
multi.impute <- studypop %>% dplyr::select(HospID, Age.Tx:DLCO.z, Hosp.RR.Median:Hosp.Min.Platelets)
multi.categorical <- studypop %>% dplyr::select(HospID, Location:Vent, Smoking, DM, RBC.PLT.First24h:Opioids, Chemo)
preproc <- preProcess(multi.impute, method = "medianImpute")
imputed.data <- predict(preproc,multi.impute)
studypop <- full_join(multi.categorical, imputed.data)
studypop[rowSums(is.na(studypop)) > 0,] # Visualize to make sure there are no NAs
# studypop <- studypop[complete.cases(studypop), ] # Drop NAs

multi.train.master <- studypop %>% dplyr::filter(Location == "RST") # Training Data 

## Estimate variable importance using penalized regression 
myControl <- trainControl(
  method = "cv", ## cross validation
  number = 10,   ## 10-fold
  summaryFunction = twoClassSummary, ## NEW
  classProbs = TRUE, # IMPORTANT
  verboseIter = FALSE,
)

# Select ARDS as the outcome of interest + include all other predictor variables. If training for a different outcome (e.g. Vent [need for IMV/NIV] or ARF.4L.O2 [need for O2 > 4L] substitute that in)
multi.train <- multi.train.master %>% dplyr::select(Tx.Type, Conditioning, ARDS.Final, Smoking, DM, RBC.PLT.First24h, Steroids.30days.PreAdmit, CARV, Pneumonia.24h, Septic.Shock.24h, Sepsis.24h, Opioids:Chemo, Pre.Hb:DLCO.z, Hosp.RR.Median:Hosp.Min.Platelets)

# Transform data (only used in the penalized regression - not used thereafter)
preProcValues <- preProcess(multi.train, method = c("center", "scale"))
train.transformed <- predict(preProcValues, multi.train)
test.transformed <- predict(preProcValues, multi.test)

# Penalized regression model
glm.model <- train(ARDS.Final ~ .,
                   multi.train,
                   metric = "ROC",
                   method = "glmnet",
                   tuneGrid = expand.grid(
                     alpha = 0:1,
                     lambda = 0:10/10),
                   trControl = myControl)
glmImp <- varImp(glm.model, scale = T) # Variable importance
plot(glmImp) # Plot variable importance

##### SCORE 

# Test score in training cohort 
multi.arf.score.training <- multi.train.master %>% mutate(LIPS.PNA = ifelse(Pneumonia.24h == "Yes", 2, 0), 
                                                 LIPS.CARV = ifelse(CARV == "Yes", 2, 0),
                                                 LIPS.Sepsis = ifelse(Sepsis.24h == "Yes", 4, 0), 
                                                 LIPS.SepticShock = ifelse(Septic.Shock.24h == "Yes", 2, 0),
                                                 LIPS.DLCO = ifelse(DLCO.z < -2.5, 1, 0), 
                                                 LIPS.FEV1 = ifelse(FEV1.z < -2.5, 1, 0),
                                                 LIPS.Tx = ifelse(Tx.Type == "Allogeneic", 2, 0),
                                                 LIPS.RIC = ifelse((Conditioning == "RIC/NMA"), 1, 0),
                                                 LIPS.HCO3 = ifelse(Hosp.Min.HCO3 < 21, 2, 0),
                                                 LIPS.AST = ifelse(Pre.AST > 60, 1, 0),
                                                 LIPS.Opioids = ifelse(Opioids == "Yes", 1, 0), 
                                                 LIPS.RR = ifelse(Hosp.RR.Median > 20, 1, 0),
                                                 LIPS.DM = ifelse(DM == "Yes", 1, 0), 
                                                 LIPS.Pre.Hb = ifelse(Pre.Hb < 8, 1, 0), 
                                                 LIPS.Pre.WBC = ifelse(Pre.WBC < 3, 1, 0), 
                                                 LIPS.Steroids = ifelse(Steroids.30days.PreAdmit == "Yes", 1, 0),
                                                 LIPS.Chemo = ifelse(Chemo == "Yes", 1, 0),
                                                 LIPS.BMT = LIPS.PNA + 
                                                   LIPS.CARV + 
                                                   LIPS.Sepsis + 
                                                   LIPS.SepticShock + 
                                                   LIPS.DLCO + 
                                                   LIPS.FEV1 + 
                                                   LIPS.Tx + 
                                                   LIPS.RIC +
                                                   LIPS.Opioids + 
                                                   LIPS.RR + 
                                                   LIPS.DM + 
                                                   LIPS.Chemo + 
                                                   LIPS.Pre.Hb + 
                                                   LIPS.Pre.WBC + 
                                                   LIPS.Steroids + 
                                                   LIPS.AST +
                                                   LIPS.HCO3)


roc.ards.training <- roc(multi.arf.score.training$ARDS.Final,
            multi.arf.score.training$LIPS.BMT, percent=FALSE,
            # arguments for ci
            ci=TRUE, boot.n=100, ci.alpha=0.9, stratified=FALSE,
            # arguments for plot
            plot=TRUE, auc.polygon=F, max.auc.polygon=F, grid=TRUE,
            print.auc=TRUE, show.thres=TRUE)
roc.vent.training <- roc(multi.arf.score.training$Vent, multi.arf.score.training$LIPS.BMT,
            ci=TRUE, boot.n=100, ci.alpha=0.9, stratified=FALSE,
            plot=TRUE, add=FALSE, print.auc=TRUE, show.thres=TRUE)  
roc.arf.training <- roc(multi.arf.score.training$ARF.4L.O2, multi.arf.score.training$LIPS.BMT,
                        ci=TRUE, boot.n=100, ci.alpha=0.9, stratified=FALSE,
                        plot=TRUE, add=FALSE, print.auc=TRUE, show.thres=TRUE)  

#######
#######
#######
# Test Final performance of the score in the Test cohort
#######
#######
#######
multi.test.master <- studypop %>% dplyr::filter(Location == "AZ" | Location == "FL") # Test Cohort

multi.arf.score.test <- multi.test.master %>% mutate(LIPS.PNA = ifelse(Pneumonia.24h == "Yes", 2, 0), 
                                                          LIPS.CARV = ifelse(CARV == "Yes", 2, 0),
                                                          LIPS.Sepsis = ifelse(Sepsis.24h == "Yes", 4, 0), 
                                                          LIPS.SepticShock = ifelse(Septic.Shock.24h == "Yes", 2, 0),
                                                          LIPS.DLCO = ifelse(DLCO.z < -2.5, 1, 0), 
                                                          LIPS.FEV1 = ifelse(FEV1.z < -2.5, 1, 0),
                                                          LIPS.Tx = ifelse(Tx.Type == "Allogeneic", 2, 0),
                                                          LIPS.RIC = ifelse((Conditioning == "RIC/NMA"), 1, 0),
                                                          LIPS.HCO3 = ifelse(Hosp.Min.HCO3 < 21, 2, 0),
                                                          LIPS.AST = ifelse(Pre.AST > 60, 1, 0),
                                                          LIPS.Opioids = ifelse(Opioids == "Yes", 1, 0), 
                                                          LIPS.RR = ifelse(Hosp.RR.Median > 20, 1, 0),
                                                          LIPS.DM = ifelse(DM == "Yes", 1, 0), 
                                                          LIPS.Pre.Hb = ifelse(Pre.Hb < 8, 1, 0), 
                                                          LIPS.Pre.WBC = ifelse(Pre.WBC < 3, 1, 0), 
                                                          LIPS.Steroids = ifelse(Steroids.30days.PreAdmit == "Yes", 1, 0),
                                                          LIPS.Chemo = ifelse(Chemo == "Yes", 1, 0),
                                                          LIPS.BMT = LIPS.PNA + 
                                                            LIPS.CARV + 
                                                            LIPS.Sepsis + 
                                                            LIPS.SepticShock + 
                                                            LIPS.DLCO + 
                                                            LIPS.FEV1 + 
                                                            LIPS.Tx + 
                                                            LIPS.RIC +
                                                            LIPS.Opioids + 
                                                            LIPS.RR + 
                                                            LIPS.DM + 
                                                            LIPS.Chemo + 
                                                            LIPS.Pre.Hb + 
                                                            LIPS.Pre.WBC + 
                                                            LIPS.Steroids + 
                                                            LIPS.AST +
                                                            LIPS.HCO3)


roc.ards.test <- roc(multi.arf.score.test$ARDS.Final,
                         multi.arf.score.test$LIPS.BMT, percent=FALSE,
                         # arguments for ci
                         ci=TRUE, boot.n=100, ci.alpha=0.9, stratified=FALSE,
                         # arguments for plot
                         plot=TRUE, auc.polygon=F, max.auc.polygon=F, grid=TRUE,
                         print.auc=TRUE, show.thres=TRUE)
roc.vent.test <- roc(multi.arf.score.test$Vent, multi.arf.score.test$LIPS.BMT,
                         ci=TRUE, boot.n=100, ci.alpha=0.9, stratified=FALSE,
                         plot=TRUE, add=FALSE, print.auc=TRUE, show.thres=TRUE)  
roc.arf.test <- roc(multi.arf.score.test$ARF.4L.O2, multi.arf.score.test$LIPS.BMT,
                        ci=TRUE, boot.n=100, ci.alpha=0.9, stratified=FALSE,
                        plot=TRUE, add=FALSE, print.auc=TRUE, show.thres=TRUE)  
