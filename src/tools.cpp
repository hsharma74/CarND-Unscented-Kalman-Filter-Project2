#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  /**
    * Calculate the RMSE here.
  **/
  VectorXd rmse(4);
  rmse << 0,0,0,0;
  
  // check that estimation vector size in on-zero
  // check that estimation and ground_truth vectors are equal in size
  assert( estimations.size() != 0 );
  assert( estimations.size() == ground_truth.size() );

  // accumulate squared residuals
  for ( int i = 0; i < estimations.size(); i++ ) { 
    VectorXd residual = estimations[i] - ground_truth[i];
    residual = residual.array() * residual.array();
    rmse += residual;
  }

  // calculate the mean of the squared residuals
  VectorXd mean(4);
  mean = rmse / estimations.size();

  // now get the square root of the mean of squared residuals
  rmse = mean.array().sqrt(); 
  
  return rmse;

}
