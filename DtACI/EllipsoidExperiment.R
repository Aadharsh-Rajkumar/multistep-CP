### ============================================================================
### Ellipsoid DtACI Experiment
### Tests the multidimensional DtACI on simulated 2D data and stock returns
### ============================================================================

source("EllipsoidDtACI.R")
library(ggplot2)
library(gridExtra)
library(dplyr)
library(cowplot)

### ============================================================================
### EXPERIMENT 1: Simulated 2D Data with Regime Shifts
### ============================================================================

set.seed(42)

# Regime 1: errors spread along the x-axis (high x-variance, low y-variance)
Sigma1 <- matrix(c(4.0, 0.5,
                    0.5, 0.3), nrow = 2, byrow = TRUE)

# Regime 2: errors spread along the y-axis (low x-variance, high y-variance)
Sigma2 <- matrix(c(0.3, -0.2,
                   -0.2, 4.0), nrow = 2, byrow = TRUE)

T_sim <- 3000
gap <- 750  # switch every 750 steps

errors_sim <- generateMultidimRegimeShift(T_sim, gap, Sigma1, Sigma2)

cat("Generated", T_sim, "simulated errors with regime switches every", gap, "steps\n")
cat("Regime 1 Sigma:\n"); print(Sigma1)
cat("Regime 2 Sigma:\n"); print(Sigma2)

### Run Ellipsoid DtACI
alpha <- 0.1
gammaGrid <- c(0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064, 0.128)

cat("\nRunning Ellipsoid DtACI on simulated data...\n")
result_sim <- ellipsoidDtACI(errors_sim, alpha, gammaGrid, sigma = 1/1000, eta = 2.72)

cat(sprintf("\nFinal coverage: %.3f (target: %.3f)\n",
            1 - mean(result_sim$errSeq), 1 - alpha))

### ============================================================================
### PLOT 1: Ellipsoids at different time snapshots
### ============================================================================

# Pick representative time steps from different regimes
snapshots <- c(200, 500, 800, 1100, 1500, 2000, 2500, 2900)
window <- 50  # show recent errors around each snapshot

ellipsePlots <- list()
for(i in seq_along(snapshots)){
  t_snap <- snapshots[i]
  regime <- ifelse(((t_snap - 1) %/% gap) %% 2 == 0, "Regime 1", "Regime 2")

  # Recent errors for scatter
  idx <- max(1, t_snap - window):t_snap
  scatter_df <- data.frame(x = errors_sim[idx, 1], y = errors_sim[idx, 2])

  # Current ellipsoid
  ell_pts <- ellipsePoints(result_sim$scaledSigmas[[t_snap]])

  # True ellipsoid for reference
  if(((t_snap - 1) %/% gap) %% 2 == 0){
    trueSigma <- scaleEllipsoid(Sigma1, alpha, 2)
  } else {
    trueSigma <- scaleEllipsoid(Sigma2, alpha, 2)
  }
  true_ell_pts <- ellipsePoints(trueSigma)

  ellipsePlots[[i]] <- ggplot() +
    geom_point(data = scatter_df, aes(x = x, y = y), alpha = 0.4, size = 1.5, color = "grey40") +
    geom_path(data = ell_pts, aes(x = x, y = y), color = "blue", linewidth = 1.2) +
    geom_path(data = true_ell_pts, aes(x = x, y = y), color = "red", linewidth = 1, linetype = "dashed") +
    coord_fixed(xlim = c(-8, 8), ylim = c(-8, 8)) +
    labs(title = sprintf("t=%d (%s)", t_snap, regime), x = "", y = "") +
    theme_bw() +
    theme(plot.title = element_text(size = 10, face = "bold"),
          axis.text = element_text(size = 8))
}

pdf("Figures/EllipsoidSnapshots.pdf", width = 16, height = 8)
grid.arrange(grobs = ellipsePlots, ncol = 4, nrow = 2,
             top = "Ellipsoid Evolution (Blue = DtACI, Red dashed = True)")
dev.off()
cat("Saved Figures/EllipsoidSnapshots.pdf\n")

### ============================================================================
### PLOT 2: Rolling Coverage Over Time
### ============================================================================

windowSize <- 200
rollingCov <- sapply(1:T_sim, function(t){
  start <- max(1, t - windowSize + 1)
  1 - mean(result_sim$errSeq[start:t])
})

covDf <- data.frame(time = 1:T_sim, coverage = rollingCov)

# Add regime switch lines
switchTimes <- seq(gap, T_sim, by = gap)

pCov <- ggplot(covDf, aes(x = time, y = coverage)) +
  geom_line(color = "blue", linewidth = 0.8) +
  geom_hline(yintercept = 1 - alpha, linetype = "dashed", color = "black", linewidth = 1) +
  geom_vline(xintercept = switchTimes, linetype = "dotted", color = "grey50", linewidth = 0.5) +
  ylim(0.5, 1.05) +
  labs(x = "Time", y = "Rolling Coverage", title = "Rolling Coverage (window=200)") +
  theme_bw() +
  theme(axis.text = element_text(size = 12), axis.title = element_text(size = 14))

### ============================================================================
### PLOT 3: Selected Gamma Over Time
### ============================================================================

gammaDf <- data.frame(time = 1:T_sim, gamma = result_sim$gammaSeq)

pGamma <- ggplot(gammaDf, aes(x = time, y = gamma)) +
  geom_line(color = "darkgreen", linewidth = 0.5, alpha = 0.6) +
  geom_vline(xintercept = switchTimes, linetype = "dotted", color = "grey50", linewidth = 0.5) +
  labs(x = "Time", y = expression(gamma[t]), title = "Selected Learning Rate Over Time") +
  theme_bw() +
  theme(axis.text = element_text(size = 12), axis.title = element_text(size = 14))

### ============================================================================
### PLOT 3.5: Adaptive Alpha Over Time
### ============================================================================
alphaDf <- data.frame(time = 1:T_sim, alphaT = result_sim$alphaSeq)

pAlpha <- ggplot(alphaDf, aes(x = time, y = alphaT)) +
  geom_line(color = "purple", linewidth = 0.8) +
  geom_hline(yintercept = alpha, linetype = "dashed", color = "black") +
  geom_vline(xintercept = switchTimes, linetype = "dotted", color = "grey50", linewidth = 0.5) +
  labs(x = "Time", y = expression(alpha[t]), title = "Shared Adaptive Alpha (controls size)") +
  theme_bw() +
  theme(axis.text = element_text(size = 12), axis.title = element_text(size = 14))

### ============================================================================
### PLOT 4: Mahalanobis Distance Over Time
### ============================================================================

mahalDf <- data.frame(time = 1:T_sim, mahal = result_sim$mahalSeq)

pMahal <- ggplot(mahalDf, aes(x = time, y = mahal)) +
  geom_point(alpha = 0.15, size = 0.5, color = "grey40") +
  geom_hline(yintercept = 1, linetype = "dashed", color = "red", linewidth = 1) +
  geom_vline(xintercept = switchTimes, linetype = "dotted", color = "grey50", linewidth = 0.5) +
  ylim(0, 10) +
  labs(x = "Time", y = "Mahalanobis Distance", title = "Mahalanobis Distance (below 1 = covered)") +
  theme_bw() +
  theme(axis.text = element_text(size = 12), axis.title = element_text(size = 14))

pdf("Figures/EllipsoidDiagnostics.pdf", width = 14, height = 16)
grid.arrange(pCov, pGamma, pAlpha, pMahal, ncol = 1)
dev.off()
cat("Saved Figures/EllipsoidDiagnostics.pdf\n")

### ============================================================================
### EXPERIMENT 2: Stock Market Returns (Multi-dimensional)
### ============================================================================

cat("\n=== EXPERIMENT 2: Stock Returns ===\n")

# Load stock data
amd <- read.csv("MajorDailyReturns/AMDDailyPrices.csv", header = TRUE)
nvidia <- read.csv("MajorDailyReturns/NividiaDailyPrices.csv", header = TRUE)

# Compute returns
pricesAMD <- rev(amd$Open)
returnsAMD <- sapply(2:length(pricesAMD), function(x){
  (pricesAMD[x] - pricesAMD[x-1]) / pricesAMD[x-1]
})

pricesNvidia <- rev(nvidia$Open)
returnsNvidia <- sapply(2:length(pricesNvidia), function(x){
  (pricesNvidia[x] - pricesNvidia[x-1]) / pricesNvidia[x-1]
})

# Align lengths (take the shorter one)
minLen <- min(length(returnsAMD), length(returnsNvidia))
returnsAMD <- tail(returnsAMD, minLen)
returnsNvidia <- tail(returnsNvidia, minLen)

# Use a simple baseline predictor: predict 0 return
# So errors = actual returns (since predicted return = 0)
stockErrors <- cbind(returnsAMD, returnsNvidia)
colnames(stockErrors) <- c("AMD", "Nvidia")

cat(sprintf("Stock returns: %d timesteps, 2 dimensions (AMD, Nvidia)\n", nrow(stockErrors)))

# Run Ellipsoid DtACI on stock returns
cat("Running Ellipsoid DtACI on stock returns...\n")
result_stocks <- ellipsoidDtACI(stockErrors, alpha, gammaGrid, sigma = 1/1000, eta = 2.72)

cat(sprintf("Final stock coverage: %.3f (target: %.3f)\n",
            1 - mean(result_stocks$errSeq), 1 - alpha))

### Stock Plots
T_stocks <- nrow(stockErrors)

# Rolling coverage
rollingCovStocks <- sapply(1:T_stocks, function(t){
  start <- max(1, t - windowSize + 1)
  1 - mean(result_stocks$errSeq[start:t])
})

# Pick some snapshot times for ellipses
stockSnapshots <- round(seq(500, T_stocks - 100, length.out = 8))
stockEllipsePlots <- list()
for(i in seq_along(stockSnapshots)){
  t_snap <- stockSnapshots[i]
  idx <- max(1, t_snap - window):t_snap
  scatter_df <- data.frame(x = stockErrors[idx, 1], y = stockErrors[idx, 2])
  ell_pts <- ellipsePoints(result_stocks$scaledSigmas[[t_snap]])

  stockEllipsePlots[[i]] <- ggplot() +
    geom_point(data = scatter_df, aes(x = x, y = y), alpha = 0.4, size = 1.5, color = "grey40") +
    geom_path(data = ell_pts, aes(x = x, y = y), color = "blue", linewidth = 1.2) +
    coord_fixed() +
    labs(title = sprintf("t=%d", t_snap), x = "AMD return", y = "Nvidia return") +
    theme_bw() +
    theme(plot.title = element_text(size = 10, face = "bold"),
          axis.text = element_text(size = 7))
}

pdf("Figures/StockEllipsoidSnapshots.pdf", width = 16, height = 8)
grid.arrange(grobs = stockEllipsePlots, ncol = 4, nrow = 2,
             top = "Stock Return Ellipsoids Over Time (AMD vs Nvidia)")
dev.off()
cat("Saved Figures/StockEllipsoidSnapshots.pdf\n")

# Diagnostics
covDfStocks <- data.frame(time = 1:T_stocks, coverage = rollingCovStocks)
gammaDfStocks <- data.frame(time = 1:T_stocks, gamma = result_stocks$gammaSeq)

pCovStocks <- ggplot(covDfStocks, aes(x = time, y = coverage)) +
  geom_line(color = "blue", linewidth = 0.8) +
  geom_hline(yintercept = 1 - alpha, linetype = "dashed", color = "black", linewidth = 1) +
  ylim(0.5, 1.05) +
  labs(x = "Time", y = "Rolling Coverage", title = "Stock Returns: Rolling Coverage") +
  theme_bw()

pGammaStocks <- ggplot(gammaDfStocks, aes(x = time, y = gamma)) +
  geom_line(color = "darkgreen", linewidth = 0.5, alpha = 0.6) +
  labs(x = "Time", y = expression(gamma[t]), title = "Stock Returns: Selected Learning Rate") +
  theme_bw()

alphaDfStocks <- data.frame(time = 1:T_stocks, alphaT = result_stocks$alphaSeq)
pAlphaStocks <- ggplot(alphaDfStocks, aes(x = time, y = alphaT)) +
  geom_line(color = "purple", linewidth = 0.8) +
  geom_hline(yintercept = alpha, linetype = "dashed", color = "black") +
  labs(x = "Time", y = expression(alpha[t]), title = "Stock Returns: Shared Adaptive Alpha (controls size)") +
  theme_bw()

pdf("Figures/StockEllipsoidDiagnostics.pdf", width = 14, height = 12)
grid.arrange(pCovStocks, pGammaStocks, pAlphaStocks, ncol = 1)
dev.off()
cat("Saved Figures/StockEllipsoidDiagnostics.pdf\n")

cat("\n=== All experiments complete! ===\n")
