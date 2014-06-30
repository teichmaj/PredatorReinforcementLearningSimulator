#include "inv_gaussian_enc.hpp"
#include "global.hpp"

#include <blitz/array.h>
#include <cmath>
#include <csignal>

using namespace blitz;


double InvGaussian::enc_freq(const Array<double,1> state)
{
double result;
result = 1.0 - exp(-((pow(state(D1)-mu(0),2.0) / (2.0*pow(sig(0),2.0))) + 
(pow(state(D2)-mu(1),2.0) / (2.0*pow(sig(1),2.0))) ) );
return result;
}

Array<double,1> InvGaussian::dd_state(const Array<double,1> state) 
{
Array<double,1> result(4);
result = 0;
double g = enc_freq(state);

 result(D1) = ((state(D1)-mu(0))/pow(sig(0),2.0)) * (1-g);
 result(D2) = ((state(D2)-mu(1))/pow(sig(1),2.0)) * (1-g);


return result;
}

Array<double,1> InvGaussian::dd_action(const Array<double,1> action)
{
Array<double,1> result(2);
result = 0;
return result;
}


bool InvGaussian::test(const Array<double,1> state,
		     const Array<double,1> action)
{
double h = 0.0001;
  bool all_pass = true;

  // state derivative only
  //
Array<double,1> _hs(4);
  Array<double,1> res_st(4);
  
  for (int i=0; i<4; i++)
    {
      _hs = 0;
      _hs(i) = h;
      Array<double,1> s1(4);
      s1 = state + _hs;
      Array<double,1> s2(4);
      s2 = state - _hs;
res_st(i) = (enc_freq(s1)- enc_freq(s2)) / (2*h);
}

 Array<double,1> dState (4);
  dState = dd_state(state);

   if ( sum(abs(res_st-dState)) > 0.001) {
     all_pass = false;
     std::cout << "InvGaussian::test() state derivative failed!\n";
   }

return all_pass;

}
