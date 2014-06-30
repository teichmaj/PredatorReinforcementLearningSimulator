#ifndef INV_GAUSSIAN_ENC_H_
#define INV_GAUSSIAN_ENC_H_

#include <blitz/array.h>

using namespace blitz;


class InvGaussian {
public:
InvGaussian (const Array<double,1> mu_,
 	    const Array<double,1> sig_):
  mu(mu_), sig(sig_) {};
  ~InvGaussian (){};

double enc_freq (const Array<double,1>);
Array<double,1> dd_state (const Array<double,1>);
Array<double,1> dd_action (const Array<double,1>);

bool test (const Array<double,1>,
	     const Array<double,1>);

private:
Array<double,1> mu;
Array<double,1> sig;

};

#endif
