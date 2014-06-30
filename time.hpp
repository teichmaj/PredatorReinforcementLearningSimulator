#ifndef TIME_H_
#define TIME_H_

#include "config.hpp"
#include "global.hpp"
#include "food_source.hpp"

#include <blitz/array.h>
using namespace blitz;


class Time {
public:
  Time (const Config *);
  ~Time ();

  Array<double,1> dot_time (const Array<double,1>,
                            const FoodSources);
  Array<double,2> dd_action (const Array<double,1>,
                             const FoodSources);
  Array<double,2> dd_state (const Array<double,1>,
                            const FoodSources);

  bool test (const Array<double,1>,
             const Array<double,1>,
             const FoodSources);

private:
  const Config * CONFIG;

};

#endif
