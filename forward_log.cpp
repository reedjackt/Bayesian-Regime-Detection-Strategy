#include <Rcpp.h>
#include <cmath> // for std::log, std::exp, std::max
using namespace Rcpp;

// "Log-Sum-Exp" Helper Function
// This prevents overflow/underflow when adding probabilities in log-space
double log_sum_exp(double a, double b) {
  if (a == -INFINITY) return b;
  if (b == -INFINITY) return a;
  double max_val = std::max(a, b);
  return max_val + std::log(1.0 + std::exp(-std::abs(a - b)));
}

// [[Rcpp::export]]
NumericMatrix forward_log_cpp(NumericVector init_probs, 
                              NumericMatrix trans_mat, 
                              NumericMatrix emission_probs) {
  
  int n_obs = emission_probs.nrow();
  int n_states = emission_probs.ncol();
  
  // we store everything as LOG probabilities
  NumericMatrix log_alpha(n_obs, n_states);
  
  // pre-compute logs of inputs to save time
  NumericVector log_init = log(init_probs);
  NumericMatrix log_trans(n_states, n_states);
  NumericMatrix log_emit(n_obs, n_states);
  
  for(int i=0; i<n_states; i++) {
    for(int j=0; j<n_states; j++) log_trans(i,j) = std::log(trans_mat(i,j));
    for(int t=0; t<n_obs; t++) log_emit(t,i) = std::log(emission_probs(t,i));
  }
  
  // initialization (t=0)
  // log(A * B) -> log(A) + log(B)
  for(int k = 0; k < n_states; k++) {
    log_alpha(0, k) = log_init[k] + log_emit(0, k);
  }
  
  // recursion (The Log-Space Loop)
  for(int t = 1; t < n_obs; t++) {
    for(int k = 0; k < n_states; k++) {
      
      // we need to sum over previous states in log-space
      // accumulator starts at -Infinity (log(0))
      double acc = -INFINITY;
      
      for(int j = 0; j < n_states; j++) {
        // log(alpha[t-1] * trans[j,k]) -> log_alpha + log_trans
        double val = log_alpha(t-1, j) + log_trans(j, k);
        
        // "Add" to accumulator using LSE trick
        acc = log_sum_exp(acc, val);
      }
      
      // update current step
      log_alpha(t, k) = acc + log_emit(t, k);
    }
  }
  
  return log_alpha;
}