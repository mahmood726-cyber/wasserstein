#!/usr/bin/env Rscript
# validate_r_ipdfromkm.R
# Cross-validate our Python Guyot IPD reconstruction against R's IPDfromKM package.
# Reads digitized curve CSVs from phase2_v2_results, feeds to IPDfromKM, compares HR.

suppressPackageStartupMessages({
  library(IPDfromKM)
  library(survival)
  library(jsonlite)
})

# --- Configuration ---
base_dir <- "C:/Users/user/Downloads/wasserstein/phase2_v2_results"

# 13 regression trials with IPD (FREEZEAF-30M added in v1.2)
trials <- list(
  list(name = "HUNTER",
       stem = "Cardiovasc electrophysiol - 2015 - HUNTER - Point-by-Point Radiofrequency Ablation Versus the Cryoballoon or a Novel",
       gt_hr = 0.53, gt_ci = c(0.25, 1.10)),
  list(name = "MILILIS-PERS",
       stem = "Cardiovasc electrophysiol - 2023 - Mililis - Radiofrequency versus cryoballoon catheter ablation in patients with",
       gt_hr = 0.96, gt_ci = c(0.50, 1.80)),
  list(name = "FIRE AND ICE",
       stem = "NEJMoa1602014",
       gt_hr = 0.96, gt_ci = c(0.76, 1.22)),
  list(name = "CIRCA-DOSE",
       stem = "andrade-et-al-cryoballoon-or-radiofrequency-ablation-for-atrial-fibrillation-assessed-by-continuous-monitoring",
       gt_hr = 1.08, gt_ci = c(0.78, 1.50)),
  list(name = "FROZEN AF",
       stem = "chun-et-al-cryoballoon-versus-laserballoon",
       gt_hr = 0.90, gt_ci = c(0.50, 1.60)),
  list(name = "CRRF-PeAF",
       stem = "ehaf451",
       gt_hr = 0.99, gt_ci = c(0.69, 1.43)),
  list(name = "HIPAF",
       stem = "euaf066",
       gt_hr = 1.47, gt_ci = c(0.70, 2.50)),
  list(name = "PVAC-CPVI",
       stem = "eut398",
       gt_hr = 0.72, gt_ci = c(0.35, 1.50)),
  list(name = "WACA-PVAC",
       stem = "euu064",
       gt_hr = 1.14, gt_ci = c(0.60, 2.10)),
  list(name = "CRAVE",
       stem = "pak-et-al-cryoballoon-versus-high-power-short-duration-radiofrequency-ablation-for-pulmonary-vein-isolation-in-patients",
       gt_hr = 0.91, gt_ci = c(0.45, 1.85)),
  list(name = "ADVENT",
       stem = "reddy-et-al-pulsed-field-vs-conventional-thermal-ablation-for-paroxysmal-atrial-fibrillation",
       gt_hr = 0.92, gt_ci = c(0.55, 1.50)),
  list(name = "FREEZEAF-30M",
       stem = "s12872-017-0566-6",
       gt_hr = 1.06, gt_ci = c(0.55, 2.00)),
  list(name = "LBRF-PERSISTENT",
       stem = "schmidt-et-al-laser-balloon-or-wide-area-circumferential-irrigated-radiofrequency-ablation-for-persistent-atrial",
       gt_hr = 0.93, gt_ci = c(0.50, 1.70))
)

# --- Helper: clean curve for IPDfromKM ---
clean_curve_for_r <- function(times, survs) {
  # Prepend (0, 1.0) if curve doesn't start at time 0 with S=1
  if (times[1] > 0 || survs[1] < 1.0) {
    times <- c(0, times)
    survs <- c(1.0, survs)
  }

  # Enforce monotonically decreasing survival
  for (i in 2:length(survs)) {
    if (survs[i] > survs[i - 1]) {
      survs[i] <- survs[i - 1]
    }
  }

  # Remove duplicate time points (keep first occurrence)
  dup <- duplicated(times)
  times <- times[!dup]
  survs <- survs[!dup]

  # Ensure sorted by time
  ord <- order(times)
  times <- times[ord]
  survs <- survs[ord]

  data.frame(time = times, surv = survs)
}

# --- Helper: generate synthetic NAR from curve ---
# IPDfromKM needs trisk/nrisk for proper interval-based reconstruction.
# Without it, everything goes into 1 interval and reconstruction is poor.
# We estimate NAR at ~10 equally spaced time points from the curve.
generate_synthetic_nar <- function(dat, n_total, n_intervals = 10) {
  t_max <- max(dat$time)
  t_min <- min(dat$time)

  # Create equally spaced time points
  trisk <- seq(t_min, t_max, length.out = n_intervals + 1)

  # Interpolate survival at each time point
  surv_at_trisk <- approx(dat$time, dat$surv, xout = trisk, rule = 2)$y

  # Estimate NAR: N_at_risk(t) ~ S(t) * N_total
  # This is an approximation (ignores cumulative censoring pattern)
  nrisk <- pmax(1, round(surv_at_trisk * n_total))

  # Ensure monotonically non-increasing
  for (i in 2:length(nrisk)) {
    if (nrisk[i] > nrisk[i - 1]) nrisk[i] <- nrisk[i - 1]
  }

  list(trisk = trisk, nrisk = nrisk)
}

# --- Main validation loop ---
cat("=" %>% rep(80) %>% paste(collapse = ""), "\n")
cat("R IPDfromKM Cross-Validation of Wasserstein KM Extractor\n")
cat("=" %>% rep(80) %>% paste(collapse = ""), "\n")
cat(sprintf("Date: %s\n", Sys.time()))
cat(sprintf("IPDfromKM version: %s\n", packageVersion("IPDfromKM")))
cat(sprintf("R version: %s\n\n", R.version.string))

results <- data.frame(
  trial = character(),
  gt_hr = numeric(),
  python_hr = numeric(),
  r_recon_hr = numeric(),
  r_recon_ci_lo = numeric(),
  r_recon_ci_hi = numeric(),
  r_pyipd_hr = numeric(),
  r_pyipd_ci_lo = numeric(),
  r_pyipd_ci_hi = numeric(),
  recon_diff_pct = numeric(),
  pyipd_diff_pct = numeric(),
  r_recon_vs_gt_pct = numeric(),
  r_pyipd_vs_gt_pct = numeric(),
  r_recon_in_gt_ci = logical(),
  r_pyipd_in_gt_ci = logical(),
  r_ipd_n = integer(),
  py_ipd_n = integer(),
  status = character(),
  stringsAsFactors = FALSE
)

for (trial in trials) {
  cat(sprintf("\n--- %s ---\n", trial$name))

  # Read curves CSV
  curves_file <- file.path(base_dir, paste0(trial$stem, "_curves.csv"))
  summary_file <- file.path(base_dir, paste0(trial$stem, "_summary.json"))

  ipd_file <- file.path(base_dir, paste0(trial$stem, "_ipd.csv"))

  if (!file.exists(curves_file) || !file.exists(summary_file)) {
    cat("  SKIP: files not found\n")
    results <- rbind(results, data.frame(
      trial = trial$name, gt_hr = trial$gt_hr,
      python_hr = NA, r_recon_hr = NA, r_recon_ci_lo = NA, r_recon_ci_hi = NA,
      r_pyipd_hr = NA, r_pyipd_ci_lo = NA, r_pyipd_ci_hi = NA,
      recon_diff_pct = NA, pyipd_diff_pct = NA,
      r_recon_vs_gt_pct = NA, r_pyipd_vs_gt_pct = NA,
      r_recon_in_gt_ci = NA, r_pyipd_in_gt_ci = NA,
      r_ipd_n = NA, py_ipd_n = NA, status = "SKIP"))
    next
  }

  # Read summary for Python's HR and arm sizes
  summ <- fromJSON(summary_file)
  py_hr <- summ$hr
  py_n1 <- summ$ipd_arm1$n_patients
  py_n2 <- summ$ipd_arm2$n_patients
  py_ipd_total <- summ$total_ipd_records

  cat(sprintf("  Python HR: %.3f, N1=%d, N2=%d, IPD=%d\n", py_hr, py_n1, py_n2, py_ipd_total))

  # Read and split curves
  curves <- read.csv(curves_file, stringsAsFactors = FALSE)
  arm_names <- unique(curves$arm)

  if (length(arm_names) != 2) {
    cat(sprintf("  SKIP: expected 2 arms, found %d\n", length(arm_names)))
    results <- rbind(results, data.frame(
      trial = trial$name, gt_hr = trial$gt_hr,
      python_hr = py_hr, r_recon_hr = NA, r_recon_ci_lo = NA, r_recon_ci_hi = NA,
      r_pyipd_hr = NA, r_pyipd_ci_lo = NA, r_pyipd_ci_hi = NA,
      recon_diff_pct = NA, pyipd_diff_pct = NA,
      r_recon_vs_gt_pct = NA, r_pyipd_vs_gt_pct = NA,
      r_recon_in_gt_ci = NA, r_pyipd_in_gt_ci = NA,
      r_ipd_n = NA, py_ipd_n = py_ipd_total, status = "SKIP_ARMS"))
    next
  }

  # Arm 1 = "treatment" (arm index 1), Arm 2 = "control" (arm index 0)
  c1 <- curves[curves$arm == "treatment", ]
  c2 <- curves[curves$arm == "control", ]

  # Clean curves (same preprocessing as Python pipeline)
  dat1 <- clean_curve_for_r(c1$time, c1$survival)
  dat2 <- clean_curve_for_r(c2$time, c2$survival)

  cat(sprintf("  Arm1 (treatment): %d points, t=[%.3f,%.3f]\n",
              nrow(dat1), min(dat1$time), max(dat1$time)))
  cat(sprintf("  Arm2 (control):   %d points, t=[%.3f,%.3f]\n",
              nrow(dat2), min(dat2$time), max(dat2$time)))

  # --- Validation 1: R IPDfromKM reconstruction → Cox HR ---
  r_recon_hr <- NA; r_recon_ci_lo <- NA; r_recon_ci_hi <- NA; r_ipd_n <- NA
  tryCatch({
    nar1 <- generate_synthetic_nar(dat1, py_n1)
    nar2 <- generate_synthetic_nar(dat2, py_n2)

    pre1 <- preprocess(dat = dat1, trisk = nar1$trisk, nrisk = nar1$nrisk, maxy = 1)
    pre2 <- preprocess(dat = dat2, trisk = nar2$trisk, nrisk = nar2$nrisk, maxy = 1)

    ipd1 <- getIPD(prep = pre1, armID = 1)
    ipd2 <- getIPD(prep = pre2, armID = 0)

    ipd_combined <- rbind(ipd1$IPD, ipd2$IPD)
    r_ipd_n <- nrow(ipd_combined)

    cox_fit <- coxph(Surv(time, status) ~ treat, data = ipd_combined)
    r_recon_hr <- exp(coef(cox_fit)["treat"])
    r_recon_ci_lo <- exp(confint(cox_fit)["treat", 1])
    r_recon_ci_hi <- exp(confint(cox_fit)["treat", 2])

    cat(sprintf("  R recon HR:  %.3f [%.3f, %.3f] (N=%d)\n",
                r_recon_hr, r_recon_ci_lo, r_recon_ci_hi, r_ipd_n))
  }, error = function(e) {
    cat(sprintf("  R recon ERROR: %s\n", conditionMessage(e)))
  })

  # --- Validation 2: Python IPD → R Cox HR ---
  r_pyipd_hr <- NA; r_pyipd_ci_lo <- NA; r_pyipd_ci_hi <- NA
  tryCatch({
    if (file.exists(ipd_file)) {
      py_ipd <- read.csv(ipd_file, stringsAsFactors = FALSE)
      # IPD format: time, event, arm (arm: 0=control, 1=treatment)
      colnames(py_ipd) <- c("time", "status", "treat")

      if (nrow(py_ipd) >= 4 && length(unique(py_ipd$treat)) == 2) {
        cox_py <- coxph(Surv(time, status) ~ treat, data = py_ipd)
        r_pyipd_hr <- exp(coef(cox_py)["treat"])
        r_pyipd_ci_lo <- exp(confint(cox_py)["treat", 1])
        r_pyipd_ci_hi <- exp(confint(cox_py)["treat", 2])

        cat(sprintf("  R on Py IPD: %.3f [%.3f, %.3f] (N=%d)\n",
                    r_pyipd_hr, r_pyipd_ci_lo, r_pyipd_ci_hi, nrow(py_ipd)))
      } else {
        cat(sprintf("  R on Py IPD: SKIP (too few records or arms)\n"))
      }
    } else {
      cat("  R on Py IPD: SKIP (no IPD file)\n")
    }
  }, error = function(e) {
    cat(sprintf("  R on Py IPD ERROR: %s\n", conditionMessage(e)))
  })

  # --- Compute metrics ---
  recon_diff <- if (!is.na(r_recon_hr)) abs(py_hr - r_recon_hr) / py_hr * 100 else NA
  pyipd_diff <- if (!is.na(r_pyipd_hr)) abs(py_hr - r_pyipd_hr) / py_hr * 100 else NA
  r_recon_vs_gt <- if (!is.na(r_recon_hr)) abs(trial$gt_hr - r_recon_hr) / trial$gt_hr * 100 else NA
  r_pyipd_vs_gt <- if (!is.na(r_pyipd_hr)) abs(trial$gt_hr - r_pyipd_hr) / trial$gt_hr * 100 else NA
  r_recon_in_ci <- if (!is.na(r_recon_hr)) r_recon_hr >= trial$gt_ci[1] && r_recon_hr <= trial$gt_ci[2] else NA
  r_pyipd_in_ci <- if (!is.na(r_pyipd_hr)) r_pyipd_hr >= trial$gt_ci[1] && r_pyipd_hr <= trial$gt_ci[2] else NA

  cat(sprintf("  Py HR vs GT:       %.1f%%\n", abs(trial$gt_hr - py_hr) / trial$gt_hr * 100))
  cat(sprintf("  R recon vs GT:     %.1f%%\n", r_recon_vs_gt))
  cat(sprintf("  R on Py IPD vs GT: %.1f%%\n", r_pyipd_vs_gt))

  results <- rbind(results, data.frame(
    trial = trial$name, gt_hr = trial$gt_hr,
    python_hr = py_hr,
    r_recon_hr = round(r_recon_hr, 3), r_recon_ci_lo = round(r_recon_ci_lo, 3),
    r_recon_ci_hi = round(r_recon_ci_hi, 3),
    r_pyipd_hr = round(r_pyipd_hr, 3), r_pyipd_ci_lo = round(r_pyipd_ci_lo, 3),
    r_pyipd_ci_hi = round(r_pyipd_ci_hi, 3),
    recon_diff_pct = round(recon_diff, 1), pyipd_diff_pct = round(pyipd_diff, 1),
    r_recon_vs_gt_pct = round(r_recon_vs_gt, 1), r_pyipd_vs_gt_pct = round(r_pyipd_vs_gt, 1),
    r_recon_in_gt_ci = r_recon_in_ci, r_pyipd_in_gt_ci = r_pyipd_in_ci,
    r_ipd_n = r_ipd_n, py_ipd_n = py_ipd_total,
    status = "PASS"))
}

# --- Summary ---
cat("\n\n")
cat("=" %>% rep(80) %>% paste(collapse = ""), "\n")
cat("SUMMARY TABLE\n")
cat("=" %>% rep(80) %>% paste(collapse = ""), "\n\n")

cat("\nA) HR Comparison: GT vs Python vs R-recon vs R-on-PyIPD\n")
cat(sprintf("%-16s %5s %6s %7s %8s\n", "Trial", "GT", "Python", "R-recon", "R-PyIPD"))
cat(paste(rep("-", 50), collapse = ""), "\n")
for (i in 1:nrow(results)) {
  r <- results[i, ]
  cat(sprintf("%-16s %5.2f %6.3f %7.3f %8.3f\n",
              r$trial, r$gt_hr, r$python_hr,
              ifelse(is.na(r$r_recon_hr), NA, r$r_recon_hr),
              ifelse(is.na(r$r_pyipd_hr), NA, r$r_pyipd_hr)))
}

cat("\nB) Error vs Ground Truth (%)\n")
cat(sprintf("%-16s %7s %7s %7s\n", "Trial", "Python", "R-recon", "R-PyIPD"))
cat(paste(rep("-", 42), collapse = ""), "\n")
for (i in 1:nrow(results)) {
  r <- results[i, ]
  py_err <- abs(r$gt_hr - r$python_hr) / r$gt_hr * 100
  cat(sprintf("%-16s %6.1f%% %6.1f%% %6.1f%%\n",
              r$trial, py_err,
              ifelse(is.na(r$r_recon_vs_gt_pct), NA, r$r_recon_vs_gt_pct),
              ifelse(is.na(r$r_pyipd_vs_gt_pct), NA, r$r_pyipd_vs_gt_pct)))
}

# Aggregate statistics
passed <- results[results$status == "PASS", ]
if (nrow(passed) > 0) {
  cat(sprintf("\n\nTrials processed: %d / %d\n", nrow(passed), nrow(results)))

  py_err <- abs(passed$gt_hr - passed$python_hr) / passed$gt_hr * 100

  cat("\n--- Python pipeline (our tool) ---\n")
  cat(sprintf("  vs GT:    median %.1f%%, mean %.1f%%\n", median(py_err), mean(py_err)))
  cat(sprintf("  in GT CI: %d / %d (%.0f%%)\n",
              sum(passed$python_hr >= sapply(trials, function(t) t$gt_ci[1]) &
                  passed$python_hr <= sapply(trials, function(t) t$gt_ci[2])),
              nrow(passed), NA))

  cat("\n--- R IPDfromKM reconstruction ---\n")
  r_recon_ok <- !is.na(passed$r_recon_vs_gt_pct)
  if (sum(r_recon_ok) > 0) {
    cat(sprintf("  vs GT:    median %.1f%%, mean %.1f%%\n",
                median(passed$r_recon_vs_gt_pct[r_recon_ok]),
                mean(passed$r_recon_vs_gt_pct[r_recon_ok])))
    cat(sprintf("  in GT CI: %d / %d (%.0f%%)\n",
                sum(passed$r_recon_in_gt_ci[r_recon_ok], na.rm = TRUE),
                sum(r_recon_ok),
                100 * sum(passed$r_recon_in_gt_ci[r_recon_ok], na.rm = TRUE) / sum(r_recon_ok)))
  }

  cat("\n--- R Cox on Python IPD (HR estimation validation) ---\n")
  r_pyipd_ok <- !is.na(passed$r_pyipd_vs_gt_pct)
  if (sum(r_pyipd_ok) > 0) {
    cat(sprintf("  vs GT:    median %.1f%%, mean %.1f%%\n",
                median(passed$r_pyipd_vs_gt_pct[r_pyipd_ok]),
                mean(passed$r_pyipd_vs_gt_pct[r_pyipd_ok])))
    cat(sprintf("  in GT CI: %d / %d (%.0f%%)\n",
                sum(passed$r_pyipd_in_gt_ci[r_pyipd_ok], na.rm = TRUE),
                sum(r_pyipd_ok),
                100 * sum(passed$r_pyipd_in_gt_ci[r_pyipd_ok], na.rm = TRUE) / sum(r_pyipd_ok)))
    cat(sprintf("  vs Py HR: median %.1f%%, mean %.1f%% (should be <5%% if log-rank agrees with Cox)\n",
                median(passed$pyipd_diff_pct[r_pyipd_ok]),
                mean(passed$pyipd_diff_pct[r_pyipd_ok])))
  }

  # Correlation
  if (sum(r_recon_ok) > 2) {
    cor_recon <- cor(passed$python_hr[r_recon_ok], passed$r_recon_hr[r_recon_ok])
    cat(sprintf("\nPython-R recon correlation: r = %.4f\n", cor_recon))
  }
  if (sum(r_pyipd_ok) > 2) {
    cor_pyipd <- cor(passed$python_hr[r_pyipd_ok], passed$r_pyipd_hr[r_pyipd_ok])
    cat(sprintf("Python-R PyIPD correlation: r = %.4f\n", cor_pyipd))
  }
}

# Save results as JSON
output_file <- file.path(dirname(base_dir), "r_ipdfromkm_validation.json")
write_json(results, output_file, pretty = TRUE, auto_unbox = TRUE)
cat(sprintf("\nResults saved to: %s\n", output_file))

cat("\nDone.\n")
