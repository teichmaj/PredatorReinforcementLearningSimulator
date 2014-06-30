#ifndef ENERGY_H_
#define ENERGY_H_


#include "config.hpp"
#include "global.hpp"
#include "food_source.hpp"

#include <blitz/array.h>
using namespace blitz;


class Energy {
public:
  Energy (const Config *);
  ~Energy ();

  double dot_energy (const Array<double,1>,
                     const Array<double,1>,
                     const FoodSources);

  Array<double,1> dd_action (const Array<double,1>);
  Array<double,1> dd_state (const Array<double,1>,
                             const FoodSources);
  
  bool test (const Array<double,1>,
             const Array<double,1>,
             const FoodSources);

private:
const Config * CONFIG;


};

#endif
