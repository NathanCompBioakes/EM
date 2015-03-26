#include <vector>
#include <string>
#include <utility>

typedef std::vector<std::pair<int, double>> histogram;

struct theta {
	theta();
	theta( double mu, double sigma, double lambda, double mixture, double divergence );
	double m_mu;
	double m_sigma;
	double m_lambda;
	double m_normal_mixture;
	double m_divergence;
};

namespace ModelHistogram {
	const double INV_SQRT_2PI = 0.3989422804014327;

	double bayes( double normal, double exp, double alpha );

	double KL_Divergence( const histogram& p, const histogram& q );

	double normal_mean_expected( const histogram& data, const histogram& mixture );

	double normal_sigma_expected( const histogram& data, const histogram& mixture );

	double exp_lambda_expected( const histogram& data, const histogram& mixture );

	void normalize( histogram& abnormal );

	double normal_pdf( const int x, const double mu, const double sigma );

	double exp_pdf( const int x, const double lambda );

	theta maximization_step( const histogram& data_set, const histogram& mixture );

	histogram expectation_step( const histogram& data_set, const theta& theta_data );

	histogram read_in( const std::string& file_name );

	histogram simulate_dist( double mu, double sigma, double lambda, const histogram& mixture );

	theta find_theta( const std::string& file_name );
} // namespace stat
