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

corr_clean <- read.csv(file="F:/BindEEG/ParticipantsData/ThetaData_filtered3sd.csv", header=TRUE, sep=",")

# converting RTs to ms
corr_clean$RT = corr_clean$RT * 1000

################################################
## Requisite for mediation analysis
################################################

# 1 
# Effect of Task on RTs

model = lmer(RT ~ Task + (1 + Task| Subject), data = corr_clean)
summary(model)
anova(model)
plot(effect('Task', model), xlab = "Task", ylab = "RT")
confint(model, parm = names(fixef(model))[-1], method = "Wald")

# 2
# Effect of theta and task on RTs

model = lmer(RT ~ mean_theta + Task + (1 + Task| Subject), data = corr_clean)
summary(model)
anova(model)
plot(effect('mean_theta', model), xlab = "mean_theta", ylab = "RT")
plot(effect('Task', model), xlab = "Task", ylab = "RT")

confint(model, parm = names(fixef(model))[-1], method = "Wald")

# 3
# Effect of Task on theta

model = lmer(mean_theta ~ Task + (1 + Task| Subject), data = corr_clean)
summary(model)
anova(model)
plot(effect('Task', model), xlab = "Task", ylab = "mean_theta", main = "Trial-By-Trial EEG")
confint(model, parm = names(fixef(model))[-1], method = "Wald")


################################################
## Mediation analysis
################################################

library(mediation)
library(lmerTest)
detach("package:lmerTest", unload=TRUE)
library(lme4)

set.seed(08022022)

corr_clean$Task <- as.numeric(as.factor(corr_clean$Task))
class(corr_clean$Task)

mediator.m <- lmer(mean_theta ~ Task + (1 + Task | Subject), data = corr_clean)
summary(mediator.m)
outcome.m <- lmer(RT ~ mean_theta + Task + (1 + Task | Subject), data = corr_clean)
summary(outcome.m)
  
mediation <- mediate(model.m = mediator.m, model.y = outcome.m, 
                     treat = "Task", mediator = "mean_theta", 
                    control.value = 1, treat.value = 2, sims = 5000)

summary(mediation)
summary(mediation)$d0
summary(mediation)$z0

plot(mediation)

# ACME = average causal mediation effect
# ADE = average direct effect







##################################################
## PLOT
##################################################

# For the plot i use a model with interaction (?)

corr_clean$Task = as.factor(corr_clean$Task)
model2 = lmer(RT ~ mean_theta * Task + (1 + Task| Subject), data = corr_clean)


set_theme(
  base = theme_classic(),
  theme.font = 'sans',
  title.color = "black",
  title.size = 3,
  title.align = "center",
  
  axis.title.color = "black",
  axis.title.size = 2,
  
  axis.textcolor = 'black',
  
  axis.linecolor = 'black',
  axis.line.size = 1,
  axis.textsize = 1.2,
  axis.tickslen = 0,
  
  
  legend.pos = "None",
  legend.size = 1.5,
  legend.color = "black",
  legend.title.size = 2,
  legend.title.color = "black",
  legend.title.face = "bold",
  legend.backgroundcol = "white",
  legend.bordercol = "white",
  # legend.item.size = 1,
  legend.item.backcol = "white",
  legend.item.bordercol = "white"
)



# dlist <- c(0.08, 2, 4, 6, 8)

dlist <- c(0.08, 5.3, 11, 16, 21) 

# Implementation
effect_impl <- as.numeric(summary(effect('mean_theta:Task', model2, xlevels = list( 'mean_theta:Task'  = dlist)))$effect[,1])
lower_impl <- as.numeric(summary(effect('mean_theta:Task', model2, xlevels = list('mean_theta:Task' = dlist)))$lower[,1])
upper_impl <- as.numeric(summary(effect('mean_theta:Task', model2, xlevels = list('mean_theta:Task' = dlist)))$upper[,1])


# Memorization
effect_memo <- as.numeric(summary(effect('mean_theta:Task', model2, xlevels = list( 'mean_theta:Task'  = dlist)))$effect[,2])
lower_memo <- as.numeric(summary(effect('mean_theta:Task', model2, xlevels = list('mean_theta:Task' = dlist)))$lower[,2])
upper_memo <- as.numeric(summary(effect('mean_theta:Task', model2, xlevels = list('mean_theta:Task' = dlist)))$upper[,2])


par(mfrow = c(1,2)) #or: layout(matrix(c(1,2),1,2))

plot(effect_impl ~ dlist, col = 'blue', lwd = 3, 
     xlim = c(0, 21), ylim = c(550, 800), type = "l",
     yaxs = "i", xaxs = "i", axes = F, 
     xlab = "Mean Theta Power (a.u.)", ylab = "RT (ms)")
lines(lower_impl ~ dlist, col = "blue", lty = 2)
lines(upper_impl ~ dlist, col = "blue", lty = 2)
axis(1, at = seq(from = 0, to = 21, by = 2))
axis(2, at = seq(from = 550, to = 850, by = 50))
title("Implementation", cex.main = 1.50)


plot(effect_memo ~ dlist, col = 'red', lwd = 3, 
     xlim = c(0, 21), ylim = c(1050, 1300), type = "l",
     yaxs = "i", xaxs = "i", axes = F, 
     xlab = "Mean Theta Power (a.u.)", ylab = "RT (ms)")
lines(lower_memo ~ dlist, col = "red", lty = 2)
lines(upper_memo ~ dlist, col = "red", lty = 2)

axis(1, at = seq(from = 0, to = 21, by = 2))
axis(2, at = seq(from = 1050, to = 1300, by = 50))

title("Memorization", cex.main = 1.50)



