#include <iostream>
#include <string>
#include "ModelHistogram.h"

int main( int argc, char* argv[] ) {
	std::string file_in{ argv[1] };
	theta best = find_theta( file_in );
	std::cout << "mu=" << best.m_mu << " sigma=" << best.m_sigma << " lambda=" <<
		best.m_lambda << " mix=" << best.m_normal_mixture
		<< " divergence=" << best.m_divergence << std::endl;
	return 0;
}
