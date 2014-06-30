#ifndef _RPROP_H_
#define _RPROP_H_


#include <blitz/array.h>
#include "config.hpp"

using namespace blitz;


class RPROP {
public:
  RPROP (const Config*);
  Array<double,1> opt(const Array<double,1>);
  bool test ();

private:
  const Config * CONFIG;
  Array<double,1> prev_dW;
  int NUM;
  double init_up;
  double eta_minus;
  double eta_plus;
  double max_up;
  double min_up;
};

#endif
