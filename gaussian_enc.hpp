#ifndef GAUSSIAN_ENC_H_
#define GAUSSIAN_ENC_H_

#include <blitz/array.h>

using namespace blitz;


class Gaussian {
public:
Gaussian (const Array<double,1> mu_,
	    const Array<double,1> sig_):
  mu(mu_), sig(sig_) {};
~Gaussian (){};

double enc_freq (const Array<double,1>);
Array<double,1> dd_state (const Array<double,1>);
Array<double,1> dd_action (const Array<double,1>);

bool test (const Array<double,1>,
	     const Array<double,1>);

  Array<double,1> location ();

private:
Array<double,1> mu;
Array<double,1> sig;

};

#endif
