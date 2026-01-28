data {
  int<lower=1> T;                   // nO. of observations (days)
  vector[T] r;                      // daily log returns
  
  // hyperparameters for priors 
  vector<lower=0>[2] alpha_calm;    // Dirichlet prior for Calm regime transitions
  vector<lower=0>[2] alpha_crisis;  // Dirichlet prior for Crisis regime transitions
  real<lower=0> mu_scale;           // scale for mean prior
  real<lower=0> sigma_scale;        // scale for volatility prior
}

parameters {
  vector[2] mu;                     // regime means
  
 
  // sigma[1] will ALWAYS be the lower volatility (Calm)
  // sigma[2] will ALWAYS be the higher volatility (Crisis)
  positive_ordered[2] sigma;        
  
  simplex[2] P[2];                  // transition probability matrix
}

transformed parameters {
  vector[2] pi;                     // stationary distribution (Long-run probabilities)
  matrix[T, 2] log_alpha;           // forward variable: log p(z_t, r_{1:t})
  matrix[T, 2] log_dens;            // pre-computed log-likelihoods

  // calculate Stationary distribution:
  // We solve pi * P = pi analytically for the 2 state case.
  // this ensures the Day 1 probability is consistent with the long-run dynamics.
  // pi_1 = (1 - P22) / ( (1 - P11) + (1 - P22) )
  pi[1] = (1 - P[2, 2]) / ((1 - P[1, 1]) + (1 - P[2, 2]));
  pi[2] = 1 - pi[1];

  // pre-calculate log likelihoods 
  // this removes the expensive normal_lpdf call from the recursive loop
  for (t in 1:T) {
    log_dens[t, 1] = normal_lpdf(r[t] | mu[1], sigma[1]);
    log_dens[t, 2] = normal_lpdf(r[t] | mu[2], sigma[2]);
  }

  // Recursive Step (forward algorithm)
  
  // Initialization (t=1) using the Stationary Distribution 'pi'
  log_alpha[1, 1] = log(pi[1]) + log_dens[1, 1];
  log_alpha[1, 2] = log(pi[2]) + log_dens[1, 2];

  // recursion (t=2 to T)
  for (t in 2:T) {
    for (j in 1:2) {
      // accumulate probability from previous states i -> current state j
      real acc[2];
      acc[1] = log_alpha[t-1, 1] + log(P[1, j]);
      acc[2] = log_alpha[t-1, 2] + log(P[2, j]);
      
      log_alpha[t, j] = log_sum_exp(acc) + log_dens[t, j];
    }
  }
}

model {
  // priors 
  mu ~ normal(0, mu_scale);

  // priors on volatilities 
  // Since sigma is ordered, the joint prior is effectively truncated
  sigma[1] ~ normal(0, sigma_scale);
  sigma[2] ~ normal(0, sigma_scale * 3); // we allow a fatter tail for crisis vol 

  // priors on transition probabilities (persitence)
  P[1] ~ dirichlet(alpha_calm);     // e.g., [10, 2] -> favors staying calm
  P[2] ~ dirichlet(alpha_crisis);   // e.g., [1, 10] -> favors staying crisis

  // likelihood
  // sum of the final forward probabilities is the marginal likelihood
  target += log_sum_exp(log_alpha[T]);
}

generated quantities {
  vector[T] prob_crisis;            // filtered
  vector[T] prob_crisis_smooth;     // smoothed
  real log_lik;                     // total log-likelihood 
  
  // log-likelihood
  log_lik = log_sum_exp(log_alpha[T]);

  // filtered probabilities (fwd only)
  for (t in 1:T) {
    prob_crisis[t] = softmax(log_alpha[t]')[2]; 
  }

  // smoothed probabilities (Forward-Backward)
  {
    matrix[T, 2] log_beta;          // bwd var: log p(r_{t+1:T} | z_t)
    
    // initialize backward algorithm at time t 
    // Beta_T = 1 (log(1) = 0) because there is no future data after T
    log_beta[T, 1] = 0;
    log_beta[T, 2] = 0;

    // backward recursion (from T-1 down to 1)
    // beta_t(i) = sum_j [ P_ij * p(r_{t+1} | j) * beta_{t+1}(j) ]
    for (t_rev in 1:(T - 1)) {
      int t = T - t_rev; // t goes T-1, T-2, ..., 1
      
      for (i in 1:2) { // for current state i
        real acc[2];
        for (j in 1:2) { // sum over next state j
          acc[j] = log(P[i, j]) + log_dens[t+1, j] + log_beta[t+1, j];
        }
        log_beta[t, i] = log_sum_exp(acc);
      }
    }

    // combine forward and backward for smoothing 
    // gamma_t(k) propto alpha_t(k) * beta_t(k)
    for (t in 1:T) {
      vector[2] log_gamma;
      log_gamma = log_alpha[t]' + log_beta[t]'; // element-wise sum in log space
      
      // normalize and store crisis probability
      prob_crisis_smooth[t] = softmax(log_gamma)[2];
    }
  }
}
