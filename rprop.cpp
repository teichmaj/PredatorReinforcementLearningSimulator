#include "rprop.hpp"
#include "global.hpp"

#include <cmath>
#include <blitz/array.h>


using namespace blitz;

double sgn(double val) {
  return (0.0 < val) - (val < 0.0);
}

BZ_DECLARE_FUNCTION(sgn)

RPROP::RPROP(const Config * CONFIG_)
{
  CONFIG = CONFIG_;
  init_up = 0.1;
  eta_minus = 0.9;
  eta_plus = 1.1;
  max_up = 100;
  min_up = 1e-8;
  NUM = 0;
}


Array<double,1> RPROP::opt(const Array<double,1> dJdw)
{
  if (NUM != dJdw.numElements())
    {
      NUM = dJdw.numElements();
      prev_dW.resize(NUM);
      prev_dW = init_up;
    }

  Array<double,1> sgnChange (NUM);
  sgnChange = sgn(dJdw) * sgn(prev_dW);

  
  Array<double,1> deltaW(NUM);
  deltaW = abs(prev_dW);


  Array<double,1>::const_iterator sgn_it;
  Array<double,1>::iterator dW_it = deltaW.begin();
  for (sgn_it = sgnChange.begin();
       sgn_it != sgnChange.end();
       ++sgn_it)
    {
      if (*sgn_it > 0) {
	*dW_it *= eta_plus;
      } else if (*sgn_it < 0) {
	*dW_it *= eta_minus;
      }
      
      if (*dW_it < min_up) {
	*dW_it = min_up;
      } else if (*dW_it > max_up) {
	*dW_it = max_up;
      }
      
      ++dW_it;
    }


  Array<double,1> result(NUM);
  result = sgn(dJdw) * deltaW;

  prev_dW = result;

  return result;
}



bool RPROP::test()
{
  bool all_pass = true;
  Array<double,1> input1(10);
  input1 = 1;
  
  Array<double,1> result(10);
  result = opt(input1);
  
  if (!all(result == init_up * eta_plus))
    {
      all_pass = false;
      std::cout << "RPROP: initializing failed\n";
      std::cout << result;
    }

  result = opt(input1);
  if (!all(result == (init_up * eta_plus * eta_plus)))
    {
      all_pass = false;
      std::cout << "RPROP: increase failed\n";
      std::cout << result;
    }

  input1(5) = -1;
  result = opt(input1);

  if (!result(0) == (init_up * eta_plus * eta_plus * eta_plus))
    {
      all_pass = false;
      std::cout << "RPROP: signum change failed! (2)\n";
      std::cout << result;
    }
  if (!result(5) == (init_up * eta_plus * eta_plus * eta_minus * -1.0))
    {
      all_pass = false;
      std::cout << "RPROP: signum change failed! (3)\n";
      std::cout << result;
    }

  return all_pass;
}
