#ifndef _BPTT_H_
#define _BPTT_H_


#include <blitz/array.h>

#include "config.hpp"
#include "NN/neural_network.hpp"
#include "global.hpp"
#include "rprop.hpp"
#include "environment.hpp"
#include "learning.hpp"

using namespace blitz;



class BPTT : public ILearning {
public:
  BPTT (const Config *, 
        ptr_environment,
        bool);
  virtual ~BPTT ();

  std::pair<double,double> learn (const  Trajectory &,
                                          const std::shared_ptr<NeuralNetwork>,
                                          const std::shared_ptr<NeuralNetwork>);
  
  bool test(std::shared_ptr<NeuralNetwork>,
                    const std::shared_ptr<NeuralNetwork>);

private:

  const Config * CONFIG;
  RPROP  * rprop;
  bool is_rprop;
  std::shared_ptr<Environment> environment;
  
};




#endif
