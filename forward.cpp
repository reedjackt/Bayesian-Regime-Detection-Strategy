#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
NumericMatrix forward_cpp(NumericVector init_probs, 
                          NumericMatrix trans_mat, 
                          NumericMatrix emission_probs) {
  
  int n_obs = emission_probs.nrow();
  int n_states = emission_probs.ncol();
  
  // create a matrix to store forward probabilities (alpha)
  NumericMatrix alpha(n_obs, n_states);
  
  // initialization (time t=0)
  // In R: alpha[1, ] <- init_probs * emission_probs[1, ]
  for(int k = 0; k < n_states; k++) {
    alpha(0, k) = init_probs[k] * emission_probs(0, k);
  }
  
  // recursion (the "Hot Loop")
  // this is where R is slow and C++ is fast
  for(int t = 1; t < n_obs; t++) {
    for(int k = 0; k < n_states; k++) {
      
      double acc = 0;
      // sum over previous states
      for(int j = 0; j < n_states; j++) {
        acc += alpha(t-1, j) * trans_mat(j, k);
      }
      
      // multiply by likelihood of current observation
      alpha(t, k) = acc * emission_probs(t, k);
    }
  }
  
  return alpha;
}