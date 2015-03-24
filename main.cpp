#include <string>
#include "em.h"

int main( int argc, char* argv[] ) {
	std::string file_in{ argv[1] };
	find_theta( file_in );
	return 0;
}
