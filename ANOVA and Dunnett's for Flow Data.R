# Create the dataframe
df <- data.frame(
  Group = rep(c("JPT/KAM1", "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F9", "F15", "F18"), each = 2),
  Value = c(-0.11915302246952, -0.161478995720721, 0.0548866270230305, -0.0973056970140926, 0.0210295634620137, 0.0543312842586232, 0.0100175632157812, -0.586936785658653, -0.0574229386987802, -0.04303742737434, 0.0504625944534731, -0.115338048288431, 0.0390581990078212, -0.197406697535133, 0.0513729217342433, -0.431500291144, 0.0657654596563914, -0.189907678480324, 0.629773188492194, 0.129924372031335, 0.362976051545883, 0.0566379768925224)
)

# Convert the 'Group' variable to a factor
df$Group <- as.factor(df$Group)

# Print the dataframe
print(df)

# Run ANOVA
anova_results <- aov(Value ~ Group, data = df)

# Print the summary
summary(anova_results)

# Install the multcomp package if it is not installed
# install.packages("multcomp")

# Load the necessary package
library(multcomp)
df$Group <- relevel(df$Group, ref = "JPT/KAM1")

# Run ANOVA
anova_results <- aov(Value ~ Group, data = df)

# Conduct Dunnett's test
dunnett_results <- glht(anova_results, linfct = mcp(Group = "Dunnett"))

# Print the summary
summary(dunnett_results)
