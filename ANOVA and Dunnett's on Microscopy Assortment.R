# Create the dataframe
df_new <- data.frame(
  Group = rep(c("JPT7/KAM1", "F3", "F5", "F7", "F15", "F18"), each = 3),
  Value = c(0.13,0.14,0.07, 0.07,0.18,0.06, -0.01,0.1,0.15, 0.06,-0.01,0.07, -0.24,-0.12,-0.07, -0.32,0.05,0.04)
)

# Convert the 'Group' variable to a factor
df_new$Group <- as.factor(df_new$Group)

# Ensure "JPT7/KAM1" is the first level of the factor
df_new$Group <- relevel(df_new$Group, ref = "JPT7/KAM1")

# Run ANOVA
anova_results_new <- aov(Value ~ Group, data = df_new)

# Print the summary of ANOVA
summary(anova_results_new)

# Load the necessary package
library(multcomp)

# Conduct Dunnett's test
dunnett_results_new <- glht(anova_results_new, linfct = mcp(Group = "Dunnett"))

# Print the summary of Dunnett's test
summary(dunnett_results_new)
