#ifndef _VGL_H_
#define _VGL_H_

#include <blitz/array.h>

#include "config.hpp"
#include "NN/neural_network.hpp"
#include "global.hpp"
#include "rprop.hpp"
#include "environment.hpp"
#include "learning.hpp"

using namespace blitz;





class VGL : public ILearning{
public:
	VGL(const Config *,
		ptr_environment,
		bool);
	
	virtual ~VGL();

	std::pair<double,double> learn (const  Trajectory &,
									const std::shared_ptr<NeuralNetwork>,
											const std::shared_ptr<NeuralNetwork>);
	
	bool test (const std::shared_ptr<NeuralNetwork>,
					   const std::shared_ptr<NeuralNetwork>);

protected:


private:

	const Config * CONFIG;
	RPROP * rprop_w;
	RPROP * rprop_z;
	bool is_rprop;
	std::shared_ptr<Environment> environment;

};













#endif // _VGL_H_
