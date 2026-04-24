### Implementation of Multidimensional DtACI for Ellipsoid Shape Adaptation
### Extends the 1D DtACI (Gradu et al., 2023) to adaptively learn ellipsoid shapes
### that achieve valid coverage in multi-dimensional prediction settings.

### ============================================================================
### Pinball loss (reused from DtACI.R)
### ============================================================================
vecZeroMax <- Vectorize(function(x){max(x,0)})
vecZeroMin <- Vectorize(function(x){min(x,0)})

pinball <- function(u, alpha){
  alpha*u - vecZeroMin(u)
}

### ============================================================================
### Mahalanobis coverage check
### Given error vector eps (d x 1) and shape matrix Sigma (d x d),
### returns TRUE if eps is inside the ellipsoid: eps^T Sigma^{-1} eps < 1
### ============================================================================
insideEllipsoid <- function(eps, Sigma){
  # Use solve for numerical stability
  mahal <- as.numeric(t(eps) %*% solve(Sigma, eps))
  return(mahal < 1)
}

### ============================================================================
### Compute Mahalanobis distance (for beta computation)
### beta_t = the quantile of how extreme the current point is
### We compute: m_t = eps^T Sigma^{-1} eps (the Mahalanobis distance squared)
### Then beta_t is conceptually the "p-value" of this score
### For the expert framework, we map this to [0,1] via:
###   beta_t = exp(-m_t/2) which is monotone in m_t and in [0,1]
### ============================================================================
computeBeta <- function(eps, Sigma, d){
  mahal <- as.numeric(t(eps) %*% solve(Sigma, eps))
  # Use chi-squared survival function: P(chi2_d > mahal)
  beta <- pchisq(mahal, df = d, lower.tail = FALSE)
  return(beta)
}

### ============================================================================
### Scale ellipsoid to achieve target coverage
### Given shape matrix Sigma and threshold alpha,
### the actual coverage ellipsoid is: { x : x^T Sigma^{-1} x < c }
### where c = qchisq(1-alpha, d) is the chi-squared quantile.
### We return the scaled matrix: c * Sigma so that the check becomes x^T S^{-1} x < 1
### ============================================================================
scaleEllipsoid <- function(Sigma, alpha, d){
  # Clamp alpha to (0.001, 0.999) to avoid NaN from qchisq
  alphaClamped <- max(0.001, min(0.999, alpha))
  scaleFactor <- qchisq(1 - alphaClamped, df = d)
  return(scaleFactor * Sigma)
}

### ============================================================================
### MAIN FUNCTION: Ellipsoid DtACI
###
### Applies the DtACI expert framework to adaptively control the learning rate
### gamma for updating the ellipsoid shape matrix Sigma.
###
### Key design — DECOUPLED shape and size:
###   - SHAPE: K experts with different gammas control how fast Sigma adapts
###   - SIZE:  One shared adaptive alpha_t controls the ellipsoid scale
###            alpha_t updates like 1D DtACI: alpha_{t+1} = alpha_t + alphaGamma*(alpha - err_t)
###            When overcovering: alpha_t increases → smaller scale → tighter fit
###            When undercovering: alpha_t decreases → bigger scale → more coverage
###
### Arguments:
###   errors     - T x d matrix of prediction errors (each row is an error vector)
###   alpha      - target miscoverage rate (e.g., 0.1 for 90% coverage)
###   gammas     - vector of candidate learning rates for the expert pool
###   sigma      - mixing parameter for stability (default 1/1000)
###   eta        - learning rate for expert weighting (default 2.72)
###   sigmaInit  - initial d x d shape matrix (NULL = auto from warm-up)
###   warmup     - number of initial points to estimate Sigma_0 (default 50)
###   alphaGamma - learning rate for the shared alpha adaptation (default 0.01)
###
### Returns a list:
###   [[1]] scaledSigmas - list of T scaled Sigma matrices (the actual ellipsoids used)
###   [[2]] rawSigmas    - list of T raw (unscaled) Sigma matrices
###   [[3]] errSeq       - vector of T coverage errors (1 = missed, 0 = covered)
###   [[4]] gammaSeq     - vector of T selected gamma values
###   [[5]] alphaSeq     - vector of T adaptive alpha values
###   [[6]] mahalSeq     - vector of T Mahalanobis distances
### ============================================================================
ellipsoidDtACI <- function(errors, alpha, gammas, sigma = 1/1000, eta = 2.72,
                           sigmaInit = NULL, warmup = 50, alphaGamma = 0.01){

  T <- nrow(errors)
  d <- ncol(errors)
  k <- length(gammas)

  # Warm-up: estimate initial Sigma from first few data points
  if(is.null(sigmaInit)){
    warmup <- min(warmup, T %/% 4)  # don't use more than 25% of data
    warmup <- max(warmup, 10)       # at least 10 points
    sigmaInit <- cov(errors[1:warmup, ])
    # Ensure positive definite with regularization
    sigmaInit <- sigmaInit + 1e-4 * diag(d)
    cat(sprintf("Warm-up: estimated initial Sigma from first %d points\n", warmup))
  }

  # Initialize all experts with the same warm-up Sigma
  expertSigmas <- replicate(k, sigmaInit, simplify = FALSE)

  # Expert weights and state
  expertWs <- rep(1, k)
  expertProbs <- rep(1/k, k)
  curExpert <- sample(1:k, 1)
  expertCumulativeLosses <- rep(0, k)

  # Shared adaptive alpha for SIZE control
  alphaT <- alpha  # starts at target

  # Output storage
  scaledSigmas <- vector("list", T)
  rawSigmas <- vector("list", T)
  errSeq <- rep(0, T)
  gammaSeq <- rep(0, T)
  alphaSeq <- rep(0, T)
  mahalSeq <- rep(0, T)

  for(t in 1:T){
    eps <- errors[t, ]

    # Current expert's Sigma
    curSigma <- expertSigmas[[curExpert]]

    # Scale the ellipsoid using the ADAPTIVE alpha
    # scaleFactor = qchisq(1 - alphaT, d)
    # When alphaT is large → scaleFactor is small → tighter ellipsoid
    # When alphaT is small → scaleFactor is large → wider ellipsoid
    alphaClamp <- max(0.001, min(0.999, alphaT))
    scaleFactor <- qchisq(1 - alphaClamp, df = d)
    scaledSigma <- scaleFactor * curSigma

    # Store outputs
    scaledSigmas[[t]] <- scaledSigma
    rawSigmas[[t]] <- curSigma
    gammaSeq[t] <- gammas[curExpert]
    alphaSeq[t] <- alphaT

    # Compute Mahalanobis distance with the scaled ellipsoid
    mahal <- tryCatch(
      as.numeric(t(eps) %*% solve(scaledSigma, eps)),
      error = function(e) Inf
    )
    mahalSeq[t] <- mahal

    # Coverage check: error = 1 means we MISSED (point outside ellipsoid)
    errSeq[t] <- as.numeric(mahal >= 1)

    # === ADAPTIVE ALPHA UPDATE (controls SIZE) ===
    # Same rule as 1D DtACI: alpha_{t+1} = alpha_t + alphaGamma * (alpha - err_t)
    # If we covered (err=0): alphaT increases → tighter next time
    # If we missed (err=1): alphaT decreases → wider next time
    alphaT <- alphaT + alphaGamma * (alpha - errSeq[t])

    # Compute coverage error for ALL experts (using current alphaT scaling)
    expertErrs <- rep(0, k)
    for(j in 1:k){
      sS <- scaleFactor * expertSigmas[[j]]
      mJ <- tryCatch(
        as.numeric(t(eps) %*% solve(sS, eps)),
        error = function(e) Inf
      )
      expertErrs[j] <- as.numeric(mJ >= 1)
    }

    # Expert loss: penalize miscoverage
    expertLosses <- abs(expertErrs - alpha)

    # Update each expert's Sigma: exponential moving average of outer products
    regTerm <- 1e-6 * diag(d)
    for(j in 1:k){
      g <- gammas[j]
      expertSigmas[[j]] <- (1 - g) * expertSigmas[[j]] + g * (eps %o% eps) + regTerm
    }

    # Update expert weights (exponential weighting with mixing)
    if(eta < Inf){
      expertBarWs <- expertWs * exp(-eta * expertLosses)
      # Guard against all-zero or NaN weights
      if(any(is.nan(expertBarWs)) || sum(expertBarWs) == 0){
        expertBarWs <- rep(1, k)
      }
      expertNextWs <- (1 - sigma) * expertBarWs / sum(expertBarWs) + sigma / k

      expertProbs <- expertNextWs / sum(expertNextWs)
      # Final NaN guard
      if(any(is.nan(expertProbs))){
        expertProbs <- rep(1/k, k)
      }
      curExpert <- sample(1:k, 1, prob = expertProbs)
      expertWs <- expertNextWs
    } else {
      expertCumulativeLosses <- expertCumulativeLosses + expertLosses
      curExpert <- which.min(expertCumulativeLosses)
    }

    if(t %% 100 == 0){
      cat(sprintf("Step %d/%d | cov: %.3f | alpha_t: %.4f | gamma: %.4f\n",
                  t, T, 1 - mean(errSeq[1:t]), alphaT, gammas[curExpert]))
    }
  }

  return(list(
    scaledSigmas = scaledSigmas,
    rawSigmas = rawSigmas,
    errSeq = errSeq,
    gammaSeq = gammaSeq,
    alphaSeq = alphaSeq,
    mahalSeq = mahalSeq
  ))
}

### ============================================================================
### HELPER: Generate 2D simulated data with covariance regime shifts
###
### Creates a sequence of 2D errors drawn from N(0, Sigma_regime)
### with periodic switches between two different covariance matrices.
###
### Arguments:
###   T       - total number of timesteps
###   gap     - number of steps between regime switches
###   Sigma1  - covariance matrix for regime 1
###   Sigma2  - covariance matrix for regime 2
###
### Returns: T x 2 matrix of error vectors
### ============================================================================
generateMultidimRegimeShift <- function(T, gap, Sigma1, Sigma2){
  d <- nrow(Sigma1)
  errors <- matrix(0, nrow = T, ncol = d)
  curRegime <- 1

  # Cholesky decompositions for sampling
  L1 <- chol(Sigma1)
  L2 <- chol(Sigma2)

  for(t in 1:T){
    if(curRegime == 1){
      errors[t, ] <- rnorm(d) %*% L1
    } else {
      errors[t, ] <- rnorm(d) %*% L2
    }
    if(t %% gap == 0){
      curRegime <- 3 - curRegime  # toggle 1 <-> 2
    }
  }
  return(errors)
}

### ============================================================================
### HELPER: Plot 2D ellipse from covariance matrix
### Returns a data frame of (x, y) points tracing the ellipse boundary
###
### Arguments:
###   Sigma  - 2x2 matrix
###   center - 2-vector center (default c(0,0))
###   npts   - number of points to trace
### ============================================================================
ellipsePoints <- function(Sigma, center = c(0, 0), npts = 100){
  angles <- seq(0, 2*pi, length.out = npts)
  unitCircle <- cbind(cos(angles), sin(angles))
  # Sigma = L %*% t(L), so ellipse = L %*% unit_circle
  L <- chol(Sigma)
  pts <- unitCircle %*% L
  pts[, 1] <- pts[, 1] + center[1]
  pts[, 2] <- pts[, 2] + center[2]
  return(data.frame(x = pts[, 1], y = pts[, 2]))
}
