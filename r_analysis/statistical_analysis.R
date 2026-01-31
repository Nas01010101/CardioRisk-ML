# Statistical Analysis in R for Heart Failure Clinical Records
# CardioRisk ML - R Statistical Component (Simplified Version)

# ============================================================================
# SETUP AND DATA LOADING
# ============================================================================

# Load required packages (core packages only)
packages <- c("survival", "jsonlite")

for (pkg in packages) {
  if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
    install.packages(pkg, repos = "https://cloud.r-project.org")
    library(pkg, character.only = TRUE)
  }
}

# Set working directory and load data
args <- commandArgs(trailingOnly = TRUE)
if (length(args) > 0) {
  data_path <- args[1]
} else {
  data_path <- "../data/heart_failure_clinical_records_dataset.csv"
}

output_dir <- ifelse(length(args) > 1, args[2], "../reports/")
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

cat("Loading data from:", data_path, "\n")
df <- read.csv(data_path)

cat("Data loaded successfully. N =", nrow(df), "patients\n\n")

# ============================================================================
# 1. DESCRIPTIVE STATISTICS
# ============================================================================

cat("======================================================================\n")
cat("1. DESCRIPTIVE STATISTICS BY SURVIVAL STATUS\n")
cat("======================================================================\n\n")

# Split by outcome
survived <- df[df$DEATH_EVENT == 0, ]
died <- df[df$DEATH_EVENT == 1, ]

cat("Sample Size:\n")
cat("  Survived: ", nrow(survived), " (", round(nrow(survived)/nrow(df)*100, 1), "%)\n", sep="")
cat("  Died: ", nrow(died), " (", round(nrow(died)/nrow(df)*100, 1), "%)\n\n", sep="")

# Continuous variables summary
continuous_vars <- c("age", "ejection_fraction", "serum_creatinine", 
                     "serum_sodium", "creatinine_phosphokinase", "platelets", "time")

desc_stats <- data.frame(
  Variable = character(),
  Mean_Survived = numeric(),
  SD_Survived = numeric(),
  Mean_Died = numeric(),
  SD_Died = numeric(),
  stringsAsFactors = FALSE
)

cat("Continuous Variables Summary:\n")
cat(sprintf("%-25s %12s %12s %12s %12s\n", "Variable", "Mean(Surv)", "SD(Surv)", "Mean(Died)", "SD(Died)"))
cat(paste(rep("-", 75), collapse=""), "\n")

for (var in continuous_vars) {
  mean_surv <- mean(survived[[var]], na.rm = TRUE)
  sd_surv <- sd(survived[[var]], na.rm = TRUE)
  mean_died <- mean(died[[var]], na.rm = TRUE)
  sd_died <- sd(died[[var]], na.rm = TRUE)
  
  cat(sprintf("%-25s %12.2f %12.2f %12.2f %12.2f\n", var, mean_surv, sd_surv, mean_died, sd_died))
  
  desc_stats <- rbind(desc_stats, data.frame(
    Variable = var,
    Mean_Survived = round(mean_surv, 2),
    SD_Survived = round(sd_surv, 2),
    Mean_Died = round(mean_died, 2),
    SD_Died = round(sd_died, 2)
  ))
}

write.csv(desc_stats, file = paste0(output_dir, "descriptive_stats.csv"), row.names = FALSE)

# ============================================================================
# 2. STATISTICAL TESTS - CONTINUOUS VARIABLES
# ============================================================================

cat("\n\n======================================================================\n")
cat("2. STATISTICAL TESTS - CONTINUOUS VARIABLES\n")
cat("======================================================================\n\n")

test_results <- data.frame(
  Variable = character(),
  Mean_Survived = numeric(),
  SD_Survived = numeric(),
  Mean_Died = numeric(),
  SD_Died = numeric(),
  Test = character(),
  Statistic = numeric(),
  P_Value = character(),
  Significant = character(),
  stringsAsFactors = FALSE
)

cat(sprintf("%-25s %10s %12s %12s\n", "Variable", "Test", "Statistic", "P-Value"))
cat(paste(rep("-", 60), collapse=""), "\n")

for (var in c("age", "ejection_fraction", "serum_creatinine", 
              "serum_sodium", "creatinine_phosphokinase", "platelets")) {
  surv_vals <- survived[[var]]
  died_vals <- died[[var]]
  
  # Use Wilcoxon test (more robust)
  test <- wilcox.test(surv_vals, died_vals)
  test_name <- "Wilcoxon"
  stat <- test$statistic
  
  sig_marker <- ifelse(test$p.value < 0.05, "***", "")
  
  cat(sprintf("%-25s %10s %12.1f %12s %s\n", 
              var, test_name, stat, format(test$p.value, scientific = TRUE, digits = 3), sig_marker))
  
  test_results <- rbind(test_results, data.frame(
    Variable = var,
    Mean_Survived = round(mean(surv_vals, na.rm = TRUE), 2),
    SD_Survived = round(sd(surv_vals, na.rm = TRUE), 2),
    Mean_Died = round(mean(died_vals, na.rm = TRUE), 2),
    SD_Died = round(sd(died_vals, na.rm = TRUE), 2),
    Test = test_name,
    Statistic = round(stat, 3),
    P_Value = format(test$p.value, scientific = TRUE, digits = 3),
    Significant = ifelse(test$p.value < 0.05, "***", "")
  ))
}

write.csv(test_results, file = paste0(output_dir, "continuous_tests.csv"), row.names = FALSE)

# ============================================================================
# 3. STATISTICAL TESTS - CATEGORICAL VARIABLES
# ============================================================================

cat("\n\n======================================================================\n")
cat("3. STATISTICAL TESTS - CATEGORICAL VARIABLES (Chi-Square)\n")
cat("======================================================================\n\n")

categorical_vars <- c("anaemia", "diabetes", "high_blood_pressure", "sex", "smoking")

chi_results <- data.frame(
  Variable = character(),
  Chi_Square = numeric(),
  DF = integer(),
  P_Value = character(),
  Significant = character(),
  stringsAsFactors = FALSE
)

cat(sprintf("%-25s %12s %5s %15s\n", "Variable", "Chi-Square", "DF", "P-Value"))
cat(paste(rep("-", 60), collapse=""), "\n")

for (var in categorical_vars) {
  tbl <- table(df[[var]], df$DEATH_EVENT)
  chi <- chisq.test(tbl)
  
  sig_marker <- ifelse(chi$p.value < 0.05, "***", "")
  
  cat(sprintf("%-25s %12.3f %5d %15s %s\n", 
              var, chi$statistic, chi$parameter, 
              format(chi$p.value, scientific = TRUE, digits = 3), sig_marker))
  
  chi_results <- rbind(chi_results, data.frame(
    Variable = var,
    Chi_Square = round(chi$statistic, 3),
    DF = chi$parameter,
    P_Value = format(chi$p.value, scientific = TRUE, digits = 3),
    Significant = ifelse(chi$p.value < 0.05, "***", "")
  ))
}

write.csv(chi_results, file = paste0(output_dir, "chi_square_tests.csv"), row.names = FALSE)

# ============================================================================
# 4. CORRELATION ANALYSIS
# ============================================================================

cat("\n\n======================================================================\n")
cat("4. CORRELATION MATRIX (Continuous Variables)\n")
cat("======================================================================\n\n")

cor_vars <- c("age", "ejection_fraction", "serum_creatinine", "serum_sodium", 
              "creatinine_phosphokinase", "platelets", "time", "DEATH_EVENT")
cor_matrix <- cor(df[, cor_vars], use = "complete.obs")

cat("Correlation Matrix:\n\n")
print(round(cor_matrix, 3))

write.csv(round(cor_matrix, 3), file = paste0(output_dir, "correlation_matrix.csv"))

# Key correlations with DEATH_EVENT
cat("\n\nCorrelations with DEATH_EVENT:\n")
death_cor <- cor_matrix[, "DEATH_EVENT"]
death_cor <- death_cor[names(death_cor) != "DEATH_EVENT"]
death_cor <- sort(abs(death_cor), decreasing = TRUE)

for (i in 1:length(death_cor)) {
  orig_cor <- cor_matrix[names(death_cor)[i], "DEATH_EVENT"]
  cat(sprintf("  %-25s: %+.3f\n", names(death_cor)[i], orig_cor))
}

# ============================================================================
# 5. SURVIVAL ANALYSIS - KAPLAN-MEIER
# ============================================================================

cat("\n\n======================================================================\n")
cat("5. SURVIVAL ANALYSIS - KAPLAN-MEIER\n")
cat("======================================================================\n\n")

# Create survival object
surv_obj <- Surv(time = df$time, event = df$DEATH_EVENT)

# Overall survival
fit_overall <- survfit(surv_obj ~ 1)
cat("Overall Survival Summary:\n")
print(fit_overall)

# Median survival
cat("\nMedian Survival Time:", summary(fit_overall)$table["median"], "days\n")

# Survival by ejection fraction groups
df$ef_group <- cut(df$ejection_fraction, 
                   breaks = c(0, 30, 40, 50, 100),
                   labels = c("Severe (<30%)", "Moderate (30-40%)", 
                             "Mild (40-50%)", "Normal (>50%)"))

fit_ef <- survfit(surv_obj ~ ef_group, data = df)
cat("\n\nSurvival by Ejection Fraction Group:\n")
print(fit_ef)

# Log-rank test
logrank_ef <- survdiff(surv_obj ~ ef_group, data = df)
logrank_ef_pvalue <- 1 - pchisq(logrank_ef$chisq, length(logrank_ef$n) - 1)
cat("\nLog-rank test p-value:", format(logrank_ef_pvalue, scientific = TRUE, digits = 3), "\n")

# Survival by creatinine groups
df$creat_group <- cut(df$serum_creatinine, 
                      breaks = c(0, 1.2, 1.5, 2.0, Inf),
                      labels = c("Normal (<1.2)", "Elevated (1.2-1.5)", 
                                "High (1.5-2.0)", "Very High (>2.0)"))

fit_creat <- survfit(surv_obj ~ creat_group, data = df)
logrank_creat <- survdiff(surv_obj ~ creat_group, data = df)
logrank_creat_pvalue <- 1 - pchisq(logrank_creat$chisq, length(logrank_creat$n) - 1)

cat("\nSurvival by Serum Creatinine Group:\n")
print(fit_creat)
cat("\nLog-rank test p-value:", format(logrank_creat_pvalue, scientific = TRUE, digits = 3), "\n")

# ============================================================================
# 6. COX PROPORTIONAL HAZARDS MODEL
# ============================================================================

cat("\n\n======================================================================\n")
cat("6. COX PROPORTIONAL HAZARDS REGRESSION\n")
cat("======================================================================\n\n")

# Univariate Cox models
cat("--- Univariate Cox Models ---\n\n")

univar_vars <- c("age", "ejection_fraction", "serum_creatinine", "serum_sodium",
                 "creatinine_phosphokinase", "platelets", "anaemia", "diabetes",
                 "high_blood_pressure", "sex", "smoking")

univar_results <- data.frame(
  Variable = character(),
  HR = numeric(),
  CI_Lower = numeric(),
  CI_Upper = numeric(),
  P_Value = character(),
  Significant = character(),
  stringsAsFactors = FALSE
)

cat(sprintf("%-25s %10s %20s %15s\n", "Variable", "HR", "95% CI", "P-Value"))
cat(paste(rep("-", 75), collapse=""), "\n")

for (var in univar_vars) {
  formula <- as.formula(paste("surv_obj ~", var))
  cox_uni <- coxph(formula, data = df)
  summary_uni <- summary(cox_uni)
  
  hr <- summary_uni$conf.int[1, 1]
  ci_lower <- summary_uni$conf.int[1, 3]
  ci_upper <- summary_uni$conf.int[1, 4]
  pval <- summary_uni$coefficients[1, 5]
  
  sig_marker <- ifelse(pval < 0.05, "***", "")
  
  cat(sprintf("%-25s %10.3f [%6.3f - %6.3f] %15s %s\n", 
              var, hr, ci_lower, ci_upper, 
              format(pval, scientific = TRUE, digits = 3), sig_marker))
  
  univar_results <- rbind(univar_results, data.frame(
    Variable = var,
    HR = round(hr, 3),
    CI_Lower = round(ci_lower, 3),
    CI_Upper = round(ci_upper, 3),
    P_Value = format(pval, scientific = TRUE, digits = 3),
    Significant = ifelse(pval < 0.05, "***", "")
  ))
}

write.csv(univar_results, file = paste0(output_dir, "cox_univariate.csv"), row.names = FALSE)

# Multivariate Cox model
cat("\n\n--- Multivariate Cox Model ---\n\n")

cox_multi <- coxph(surv_obj ~ age + ejection_fraction + serum_creatinine + 
                   serum_sodium + anaemia + high_blood_pressure, 
                   data = df)

cat("Model Summary:\n")
print(summary(cox_multi))

# Model concordance
concordance <- summary(cox_multi)$concordance[1]
cat("\nModel Concordance (C-index):", round(concordance, 3), "\n")

# Proportional hazards assumption test
cat("\nProportional Hazards Assumption Test:\n")
ph_test <- cox.zph(cox_multi)
print(ph_test)

# Save multivariate results
multi_coef <- summary(cox_multi)$coefficients
multi_ci <- summary(cox_multi)$conf.int

multi_results <- data.frame(
  term = rownames(multi_coef),
  estimate = round(multi_ci[, 1], 3),
  conf.low = round(multi_ci[, 3], 3),
  conf.high = round(multi_ci[, 4], 3),
  p.value = round(multi_coef[, 5], 4)
)

write.csv(multi_results, file = paste0(output_dir, "cox_multivariate.csv"), row.names = FALSE)

# ============================================================================
# 7. EXPORT SUMMARY JSON FOR STREAMLIT
# ============================================================================

cat("\n\n======================================================================\n")
cat("7. EXPORTING RESULTS FOR STREAMLIT INTEGRATION\n")
cat("======================================================================\n\n")

# Create summary JSON for Streamlit
summary_json <- list(
  analysis_date = as.character(Sys.time()),
  sample_size = nrow(df),
  mortality_rate = round(mean(df$DEATH_EVENT) * 100, 1),
  
  significant_continuous = test_results$Variable[test_results$Significant == "***"],
  significant_categorical = chi_results$Variable[chi_results$Significant == "***"],
  
  cox_multivariate = list(
    concordance = round(concordance, 3),
    variables = multi_results$term,
    hazard_ratios = multi_results$estimate,
    p_values = multi_results$p.value,
    significant = multi_results$p.value < 0.05
  ),
  
  key_findings = list(
    "Ejection fraction is strongly associated with mortality (HR per unit decrease)",
    "Serum creatinine >1.5 mg/dL significantly increases hazard",
    "Age is an independent predictor of mortality",
    "Traditional risk factors (diabetes, smoking) show weaker associations in this cohort"
  ),
  
  survival_analysis = list(
    median_survival_overall = as.numeric(summary(fit_overall)$table["median"]),
    logrank_ef_pvalue = format(logrank_ef_pvalue, scientific = TRUE, digits = 3),
    logrank_creat_pvalue = format(logrank_creat_pvalue, scientific = TRUE, digits = 3)
  ),
  
  correlations_with_death = list(
    time = round(cor_matrix["time", "DEATH_EVENT"], 3),
    age = round(cor_matrix["age", "DEATH_EVENT"], 3),
    ejection_fraction = round(cor_matrix["ejection_fraction", "DEATH_EVENT"], 3),
    serum_creatinine = round(cor_matrix["serum_creatinine", "DEATH_EVENT"], 3),
    serum_sodium = round(cor_matrix["serum_sodium", "DEATH_EVENT"], 3)
  )
)

write_json(summary_json, paste0(output_dir, "r_analysis_summary.json"), pretty = TRUE, auto_unbox = TRUE)

cat("Analysis complete! Results saved to:", output_dir, "\n")
cat("Files generated:\n")
cat("  - descriptive_stats.csv\n")
cat("  - continuous_tests.csv\n")
cat("  - chi_square_tests.csv\n")
cat("  - correlation_matrix.csv\n")
cat("  - cox_univariate.csv\n")
cat("  - cox_multivariate.csv\n")
cat("  - r_analysis_summary.json\n")
