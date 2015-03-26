#include <iostream>
#include <string>
#include <stdexcept>
#include "ModelHistogram.h"

int main( int argc, char* argv[] ) {
	if ( argc != 2 ) {
		throw std::runtime_error( "./ModelHistogram file_name" );
	}
	std::string file_in{ argv[1] };
	theta best = ModelHistogram::find_theta( file_in );
	std::cout << "mu=" << best.m_mu << " sigma=" << best.m_sigma << " lambda=" <<
		best.m_lambda << " mix=" << best.m_normal_mixture
		<< " divergence=" << best.m_divergence << std::endl;
	return 0;
}
