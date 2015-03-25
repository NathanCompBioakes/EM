#include <iostream>
#include <fstream>
#include <algorithm>
#include <stdexcept>
#include <utility>
#include <string>
#include <vector>
#include <random>
#include "math.h"
#include "ModelHistohram.h"
#include <iostream>

typedef std::vector<std::pair<int, double>> histogram;

theta::theta( double mu, double sigma, double lambda, double mixture, double divergence ) :
	m_mu{ mu },
	m_sigma{ sigma },
	m_lambda{ lambda },
	m_normal_mixture{ mixture },
	m_divergence{ divergence }
{}

theta::theta() :
	m_normal_mixture{ 0.75 },
	m_divergence{ 1000 } {

	std::random_device rd;
	std::uniform_real_distribution<double> distribution( 0.0, 1.0 );
	m_mu = distribution( rd );
	m_sigma = distribution( rd );
	m_lambda = distribution( rd );
}

double normal_pdf( const int x, const double mu, const double sigma ) {
	const double inv_sqrt_2pi = 0.3989422804014327;
	double a = ( x - mu ) / sigma;
	double pdf = ( inv_sqrt_2pi / sigma ) * std::exp( -0.5f * a * a );
	return pdf;
}

double exp_pdf( const int x, const double lambda ) {
	return lambda * std::exp( -1 * lambda * x );
}

histogram read_in( const std::string& file_name ) {
	try {
		std::ifstream infile( file_name );
		if ( !infile ) throw std::runtime_error("Bad filename");
		histogram data_set;
		int x;
		double y;
		while ( infile >> x >> y ) {
			if ( y != 0 ) {
				data_set.emplace_back( std::pair<int, double>{ x , y } );
			}
		}
		return data_set;
	} catch( std::exception& e ) {
		throw std::runtime_error( "Bad filename" );
	}
}

double bayes( double normal, double exp, double alpha ) {
	if ( fabs( alpha*normal - exp*( 1 - alpha ) ) < 0.00000000000001 ) {
		return 0.5;
	}
	return alpha*normal/( alpha*normal + (1-alpha)*exp );
}

void normalize( histogram& abnormal ) {
	double total = std::accumulate( abnormal.begin(), abnormal.end(), static_cast<double>( 0 ),
		[]( double total, std::pair<int, double> y){ return total + y.second; } );
	std::transform( abnormal.begin(), abnormal.end(), abnormal.begin(), [total]( std::pair<int, double>& x ) {
		x.second = x.second/total;
		return x; } );
}

double KL_Divergence( const histogram& real, const histogram& model ) {
	if ( real.size() == model.size() ) {
		double value = 0;
		histogram KL_real( real );
		double total = std::accumulate( real.begin(), real.end(), static_cast<double>( 0 ),
			[]( double total, std::pair<int, double> y){ return total + y.second; } );
		std::transform( KL_real.begin(), KL_real.end(), KL_real.begin(), [total]( std::pair<int, double>& x ) {
			x.second = x.second/total;
			return x; } );
		for ( size_t i = 0; i < real.size(); i++ ) {
			if ( real[i].second == 0 || model[i].second == 0 ) {
				continue;
			} else {
				value += KL_real[i].second*log( KL_real[i].second / model[i].second );

				// value += - KL_real[i].second*log( KL_real[i].second / model[i].second ) + log( model[i].second );
				// ***EM maximizes the difference between
				// log-likelihood function of theta given x:
				// log( L(theta|x) ) = log( P(x|theta) )
				// and
				// Kullback–Leibler divergence***
			}
		}
		return value;
	} else {
		throw std::runtime_error( "KL_divergence: q and p are of differing lengths" );
	}
}

double normal_mean_expected( const histogram& data, const histogram& mixture ) {
	double mu = 0;
	double mix = 0;
	for ( size_t i = 0; i < data.size(); i++ ) {
		mu += data[i].first*data[i].second*mixture[i].second;
		mix += data[i].second*mixture[i].second;
	}
	mu /= mix;
	return mu;
}

double normal_sigma_expected( const histogram& data, const histogram& mixture ) {
	double mu = 0;
	double mix = 0;
	for ( size_t i = 0; i < data.size(); i++ ) {
		mu += data[i].first*data[i].second*mixture[i].second;
		mix += data[i].second*mixture[i].second;
	}
	mu /= mix;

	double sigma = 0;
	for ( size_t i = 0; i < data.size(); i++ ) {
		sigma += pow( data[i].first - mu, 2 )*data[i].second*mixture[i].second;
	}
	sigma /= mix;
	sigma = sqrt( sigma );
	return sigma;
}

double exp_lambda_expected( const histogram& data, const histogram& mixture ) {
	double lambda = 0;
	double mix = 0;
	for ( size_t i = 0; i < data.size(); i++ ) {
		lambda += data[i].first*data[i].second*( 1 - mixture[i].second );
		mix += data[i].second*( 1 - mixture[i].second );
	}
	lambda /= mix;
	lambda = 1/lambda;
	return lambda;
}

histogram simulate_dist( double mu, double sigma, double lambda, const histogram& mixture ) {
	histogram simulation;
	for ( size_t i = 0; i < mixture.size(); i++ ) {
		simulation.emplace_back( std::pair<int, double> { mixture[i].first, mixture[i].second*normal_pdf( mixture[i].first, mu, sigma ) +
				( 1-mixture[i].second )*exp_pdf( mixture[i].first, lambda ) } );
	}
	normalize( simulation );
	return simulation;
}

theta maximization_step( const histogram& data_set, const histogram& mixture ) {
	double mu = normal_mean_expected( data_set, mixture );
	double sigma = normal_sigma_expected( data_set, mixture );
	double lambda = exp_lambda_expected( data_set, mixture );
	double mix = 0;
	for ( size_t i = 0; i < data_set.size(); i++ ) {
		mix += data_set[i].second*mixture[i].second;
	}
	histogram simulation = simulate_dist( mu, sigma, lambda, mixture );
	double divergence = KL_Divergence( data_set, simulation );
	return theta{ mu, sigma, lambda, mix, divergence };
}

histogram expectation_step( const histogram& data_set, const theta& theta_data ) {
	histogram mixture;
	for ( size_t i = 0; i < data_set.size(); i++ ) {
		double normal = normal_pdf( data_set[i].first, theta_data.m_mu, theta_data.m_sigma );
		double exp = exp_pdf( data_set[i].first, theta_data.m_lambda );
		mixture.emplace_back( std::pair<int, double>{ data_set[i].first, bayes( normal, exp, theta_data.m_normal_mixture ) } );
	}
	return mixture;
}

void find_theta( const std::string& file_name ) {
	histogram data_set = read_in( file_name );
	normalize( data_set );
	theta theta_best;

	int count = 100;
	while ( count-- ) {
		theta theta_data;
		theta new_theta;

		theta_data.m_mu *= data_set.size();
		theta_data.m_sigma *= data_set.size();

		bool converged = false;
		while ( !converged ) {
			histogram mixture = expectation_step( data_set , theta_data );
			new_theta = maximization_step( data_set, mixture );
			double dive = fabs( theta_data.m_divergence - new_theta.m_divergence );
			if ( dive != dive ) {
				throw std::runtime_error( "divergence nan" );
			}
			if ( fabs( theta_data.m_divergence - new_theta.m_divergence ) < 0.00001 ) {
				converged = true;
				if ( theta_best.m_divergence > new_theta.m_divergence ) {
					theta_best = new_theta;
				}
			}
			theta_data = new_theta;
		}
	}
	std::cout << "mu=" << theta_best.m_mu << " sigma=" << theta_best.m_sigma << " lambda=" << theta_best.m_lambda << " mix=" << theta_best.m_normal_mixture
		<< " divergence=" << theta_best.m_divergence << std::endl;
}
