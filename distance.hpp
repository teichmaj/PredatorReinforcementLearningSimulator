#ifndef DISTANCE_H_
#define DISTANCE_H_

#include "config.hpp"

#include <blitz/array.h>
using namespace blitz;

class Distance {
public:
  Distance (const Config *);
  ~Distance ();

  Array<double,1> dot_distance(const Array<double,1>,
                               const Array<double,1>);
  Array<double,2> dd_action(const Array<double,1>);
  Array<double,2> dd_state(const Array<double,1>);

  bool test (const Array<double,1>,
             const Array<double,1>);

  double dist_to_home (const Array<double,1>);
  Array<double,1> dd_state_dist_to_home(const Array<double,1>);

private:
  const Config * CONFIG;

  
  Array<double,1> _locomotion(const Array<double,1>);
  Array<double,1> _dd_locomotion(const Array<double,1>);

};

#endif
