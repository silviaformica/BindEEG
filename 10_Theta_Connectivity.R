library(lme4) # for the analysis
library(lmerTest)# to get p-value estimations that are not part of the standard lme4 packages
library(sjPlot)
library(sjmisc)
library(effects)
library(see)
library(ggplot2)
library(tidyverse)
library(DHARMa)
library(ggeffects)
library(raincloudplots)
library(cowplot)
library(dplyr)
library(readr)
library(forcats)
library(plyr)

options(contrasts = c("contr.sum", "contr.poly"))

data <- read.csv(file="D:/BindEEG/ParticipantsData/wPLI_db.csv", header=TRUE, sep=",")


# Modelling Hand

model_hand = lmer(hand ~ Task*Laterality + (1 + Task | Subject), data = data)

summary(model_hand)
anova(model_hand)
plot(effect('Task', model_hand), xlab = "Task", ylab = "hand", main = "Trial-By-Trial EEG")
plot(effect('Task:Laterality', model_hand), xlab = "Laterality", ylab = "hand", main = "Trial-By-Trial EEG")

confint(model_hand, parm = names(fixef(model_hand))[-1], method = "Wald")



# Modelling Visual

model_visual = lmer(visual ~ Task*Laterality + (1 + Task | Subject), data = data)

summary(model_visual)
anova(model_visual)
plot(effect('Task', model_visual), xlab = "Task", ylab = "hand", main = "Trial-By-Trial EEG")
plot(effect('Task:Laterality', model_visual), xlab = "Laterality", ylab = "visual", main = "Trial-By-Trial EEG")

confint(model_visual, parm = names(fixef(model_visual))[-1], method = "Wald")






