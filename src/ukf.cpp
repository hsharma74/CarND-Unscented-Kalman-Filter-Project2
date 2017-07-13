#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 0.5;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.5;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  /**
  TODO:

  Complete the initialization. See ukf.h for other member properties.

  Hint: one or more values initialized above might be wildly off...
  */

  is_initialized_ = false;
  n_x_            = 5;
  n_aug_          = 7;
  lambda_         = 3 - n_x_;

  Xsig_pred_ = MatrixXd( n_x_, (2 * n_aug_ + 1) ); 
  Xsig_pred_.fill(0.0);
  weights_   = VectorXd( 2 * n_aug_ + 1 );

  // initialize weights
  weights_.fill(0.0);
  weights_(0) = lambda_ / (lambda_ + n_aug_);
  for ( int i = 1; i < (2 * n_aug_ + 1); i++ ) {
     weights_(i) = 0.5 / (lambda_ + n_aug_);
  }

  //cout << "weights: " << endl << weights_ << endl;

}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Make sure you switch between lidar and radar
  measurements.
  */
  if ( !is_initialized_ ) {
     // get and transform the radar measurements
     if ( meas_package.sensor_type_ == MeasurementPackage::RADAR ) {

        double rho = meas_package.raw_measurements_[0];
        double phi = meas_package.raw_measurements_[1];
        x_ << rho * cos(phi), rho * sin(phi), 0.0, 0.0, 0.0;

     } else if ( meas_package.sensor_type_ == MeasurementPackage::LASER ) {
        // process the lidar data
        x_ << meas_package.raw_measurements_[0], meas_package.raw_measurements_[1], 0.0, 0.0, 0.0;
    
     }

     // initialize the state covariance matrix
     P_ <<  0.5,    0,    0,    0,   0,
              0,  0.5,    0,    0,   0,
              0,    0,    9,    0,   0,
              0,    0,    0,    9,   0,
              0,    0,    0,    0,   0; 

     // initialize the timestamp
     time_us_ = meas_package.timestamp_ ;

     // done initializing, so no need to predict or update in this step
     is_initialized_ = true;
     return;
  }  // if ( !is_initialized_ )

  // now process input data when it is not the first measurement
  double delta_t = (meas_package.timestamp_ - time_us_) / 1000000.0;
  time_us_       = meas_package.timestamp_ ;         

  /******************************************************************
   *   Prediction step:  calculate sigma points; predict x & P 
   *****************************************************************/
   Prediction( delta_t );


  /******************************************************************
   *   Update step: Radar and Lidar 
   *****************************************************************/

   // update state x and covariance matrix P for Radar and Lidar measurements
   if ( meas_package.sensor_type_ == MeasurementPackage::RADAR ) {
      UpdateRadar( meas_package ); 
   } else {                              
      UpdateLidar( meas_package );
   }
        
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  /**
  TODO:

  Complete this function! Estimate the object's location. Modify the state
  vector, x_. Predict sigma points, the state, and the state covariance matrix.
  */

  // create the augmented mean vector
  VectorXd x_aug = VectorXd( n_aug_ ); 

  // create the augmented covariance matrix
  MatrixXd P_aug = MatrixXd( n_aug_, n_aug_ );

  // create sigma points matrix
  MatrixXd Xsig_aug = MatrixXd( n_aug_, 2 * n_aug_ + 1 );
  
  // create augmented mean state
  x_aug.head( n_x_ ) = x_ ;
  // set the mean value for the noise; the mean is 0.0 as noise is ~N(0,sigma)
  x_aug( n_x_ )      = 0.0;
  x_aug( n_x_ + 1 )  = 0.0;

  // create the augmented covariance matrix
  P_aug.fill(0.0);
  P_aug.topLeftCorner( n_x_, n_x_ ) = P_ ;
  P_aug( n_x_, n_x_ )               = std_a_ * std_a_ ;
  P_aug( n_x_+1, n_x_+1 )           = std_yawdd_ * std_yawdd_ ;

  // create the square root matrix
  MatrixXd L = P_aug.llt().matrixL();

  // create the augmented sigma points
  const double sqrt_lambda = sqrt( lambda_ + n_aug_ );
  Xsig_aug.col(0) = x_aug ;
  for ( int i = 0; i < L.cols(); i++ ) {
     Xsig_aug.col(i+1)        =  x_aug + L.col(i) * sqrt_lambda ;
     Xsig_aug.col(i+1+n_aug_)  =  x_aug - L.col(i) * sqrt_lambda ;
  }

  // predict sigma points 
  for ( int i = 0; i < Xsig_aug.cols(); i++ ) {
     Xsig_pred_.col(i) = CalculateNextState( Xsig_aug.col(i), delta_t );  
  } 

  // predict state mean
  x_.fill(0.0);
  for ( int i = 0; i < weights_.size(); i++ ) {
     x_ += weights_(i) * Xsig_pred_.col(i);
  }

  // predict state covariance matrix
  P_.fill(0.0);
  for ( int i = 0; i < (2 * n_aug_ + 1); i++ ) {
     VectorXd x_diff = Xsig_pred_.col(i) - x_;
     // normalize the angles to be between -PI and +PI
     while ( x_diff(3) >  M_PI ) x_diff(3) -= 2.0 * M_PI;
     while ( x_diff(3) < -M_PI ) x_diff(3) += 2.0 * M_PI;
     P_ += weights_(i) * x_diff * x_diff.transpose();
  }

}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the lidar NIS.
  */

  // set the measurement dimension for lidar. Since lidar can only 
  // measure position, so dimension is 2
  int n_z = 2;

  // create matrix for sigma points in the measurement space
  MatrixXd Zsig = MatrixXd( n_z, 2 * n_aug_ + 1 ); 
  
  // create predicted measurement vector
  VectorXd z_pred = VectorXd( n_z );
  
  // measurement covariance matrix S
  MatrixXd S = MatrixXd( n_z, n_z );

  // transform sigma points into measurement space
  for ( int i = 0; i < (2 * n_aug_ + 1); i++ ) {
     double px = Xsig_pred_(0,i);  
     double py = Xsig_pred_(1,i);  
     Zsig(0,i) = px;
     Zsig(1,i) = py;
  }

  // calculate mean predicted measurement
  z_pred.fill(0.0);
  for ( int i = 0; i < (2 * n_aug_ + 1); i++ ) {
     z_pred += weights_(i) * Zsig.col(i);
  }

  // calculate measurement covariance matrix
  S.fill(0.0);
  for ( int i = 0; i < (2 * n_aug_ + 1); i++ ) {
     VectorXd z_diff = Zsig.col(i) - z_pred;
     // normalize angles to be between -PI and +PI
     while ( z_diff(1) >  M_PI ) z_diff(1) -= 2.0 * M_PI; 
     while ( z_diff(1) < -M_PI ) z_diff(1) += 2.0 * M_PI; 
     S += weights_(i) * z_diff * z_diff.transpose();
  }

  // define and add the lidar noise covariance matrix
  MatrixXd R = MatrixXd( n_z, n_z );
  R.fill(0.0);
  R(0,0) = std_laspx_ * std_laspx_;
  R(1,1) = std_laspy_ * std_laspy_;
  
  S += R;

  // calculate the cross-correlation matrix 
  MatrixXd Tc = MatrixXd( n_x_, n_z );
  Tc.fill(0.0);
  for ( int i = 0; i < (2 * n_aug_ + 1); i++ ) {
     VectorXd z_diff = Zsig.col(i) - z_pred;
     VectorXd x_diff = Xsig_pred_.col(i) - x_;
     // normalize angles to be between -PI and +PI
     while ( z_diff(1) >  M_PI ) z_diff(1) -= 2.0 * M_PI;
     while ( z_diff(1) < -M_PI ) z_diff(1) += 2.0 * M_PI;
     while ( x_diff(3) >  M_PI ) x_diff(3) -= 2.0 * M_PI;
     while ( x_diff(3) < -M_PI ) x_diff(3) += 2.0 * M_PI;
     Tc += weights_(i) *  x_diff * z_diff.transpose();         
  }

  // calculate the Kalman gain
  MatrixXd K = Tc * S.inverse();

  // get the current measurement from the lidar
  VectorXd z = VectorXd( n_z );
  z = meas_package.raw_measurements_;

  VectorXd z_diff = z - z_pred;
  // normalize angles to be between -PI and +PI
  while ( z_diff(1) >  M_PI ) z_diff(1) -= 2.0 * M_PI;
  while ( z_diff(1) < -M_PI ) z_diff(1) += 2.0 * M_PI;

  // update the state mean and covariance matrix
  x_ += K * ( z - z_pred );
  P_ -= K * S * K.transpose(); 

}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the radar NIS.
  */
  int n_z = 3;

  // create matrix for sigma points in the measurement space
  MatrixXd Zsig = MatrixXd( n_z, 2 * n_aug_ + 1 );

  // create predicted measurement vector
  VectorXd z_pred = VectorXd( n_z );
  
  // measurement covariance matrix S
  MatrixXd S = MatrixXd( n_z, n_z );

  // transform sigma points into measurement space
  for ( int i = 0; i < (2 * n_aug_ + 1); i++ ) {
     double px    = Xsig_pred_(0,i);
     double py    = Xsig_pred_(1,i);
     double speed = Xsig_pred_(2,i);
     double psi   = Xsig_pred_(3,i);

     // check for division_by_zero
     if ( fabs(px) < 0.0001 ) { px = 0.0001; }
     double rho = sqrt( px * px + py * py );
     if ( fabs(rho) < 0.0001 ) { rho = 0.0001; }

     Zsig(0,i) = rho;
     Zsig(1,i) = atan2( py, px );
     Zsig(2,i) = ( px * cos(psi) + py * sin(psi) ) * speed / rho;
  }

  // calculate mean predicted measurement
  z_pred.fill(0.0);
  for ( int i = 0; i < (2 * n_aug_ + 1);  i++ ) {
     z_pred += weights_(i) * Zsig.col(i);
  }

  // calculate measurement covariance matrix
  S.fill(0.0);
  for ( int i = 0; i < (2 * n_aug_ + 1); i++ ) {
     VectorXd z_diff = Zsig.col(i) - z_pred;
     // normalize the angles to be between -PI and +PI
     while ( z_diff(1) >  M_PI ) z_diff(1) -= 2.0 * M_PI;
     while ( z_diff(1) < -M_PI ) z_diff(1) += 2.0 * M_PI;
     S += weights_(i) *  z_diff * z_diff.transpose();
  }
  

  // define and add the measurement noise covariance matrix
  MatrixXd R = MatrixXd( n_z, n_z );
  R.fill(0.0);
  R(0,0) = std_radr_ * std_radr_;
  R(1,1) = std_radphi_ * std_radphi_;
  R(2,2) = std_radrd_ * std_radrd_;

  S += R;
 
  // create the cross-correlation matrix
  MatrixXd Tc = MatrixXd( n_x_, n_z ); 
  Tc.fill(0.0);
  for ( int i = 0; i < (2 * n_aug_ + 1); i++ ) {
     VectorXd z_diff = Zsig.col(i) - z_pred;
     VectorXd x_diff = Xsig_pred_.col(i) - x_;
     // normalize the angles to be between -PI and +PI
     while ( z_diff(1) >  M_PI ) z_diff(1) -= 2.0 * M_PI;
     while ( z_diff(1) < -M_PI ) z_diff(1) += 2.0 * M_PI;
     while ( x_diff(3) >  M_PI ) x_diff(3) -= 2.0 * M_PI;
     while ( x_diff(3) < -M_PI ) x_diff(3) += 2.0 * M_PI;
     Tc += weights_(i) * x_diff * z_diff.transpose();
  }

  // calculate Kalman gain
  MatrixXd K = Tc * S.inverse();

  // get the current measurement
  VectorXd z = VectorXd( n_z );
  z = meas_package.raw_measurements_;

  VectorXd z_diff = z - z_pred;
  // normalize angles to be between -PI and +PI
  while ( z_diff(1) >  M_PI ) z_diff(1) -= 2.0 * M_PI;
  while ( z_diff(1) < -M_PI ) z_diff(1) += 2.0 * M_PI;
  
  // update the state and covariance matrix
  x_ += K * z_diff;
  P_ -= K * S * K.transpose();     

}


VectorXd UKF::CalculateNextState( const VectorXd& xx, const double delta_t ) {
  VectorXd vec1 = VectorXd( xx.size() - 2 );
  VectorXd vec2 = VectorXd( xx.size() - 2 );

  double si     = xx(3); 
  double si_dot = xx(4); 
  double nu_a   = xx(5); 
  double nu_si  = xx(6); 
  double speed  = xx(2); 

  double dt2 = delta_t * delta_t;

  vec2(0) = 0.5 * dt2 * cos(si) * nu_a;
  vec2(1) = 0.5 * dt2 * sin(si) * nu_a;
  vec2(2) = delta_t * nu_a;
  vec2(3) = 0.5 * dt2 * nu_si;
  vec2(4) = delta_t * nu_si;

  if ( fabs(si_dot) > 0.0001 ) {
      vec1(0) = (speed / si_dot) * (  sin(si + si_dot * delta_t) - sin(si) );
      vec1(1) = (speed / si_dot) * ( -cos(si + si_dot * delta_t) + cos(si) );
      vec1(2) = 0;
      vec1(3) = si_dot * delta_t;
      vec1(4) = 0;  
   } else {
      vec1(0) = speed * cos(si) * delta_t;
      vec1(1) = speed * sin(si) * delta_t;
      vec1(2) = 0;
      vec1(3) = 0;
      vec1(4) = 0;  
   }   

   for ( int i = 0; i < 5; i++ ) { 
      vec1(i) = xx(i) + vec1(i) + vec2(i);
   }   
   return vec1; 
}
       
 
