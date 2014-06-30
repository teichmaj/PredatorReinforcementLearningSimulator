#ifndef ENVIRONMENT_H_
#define ENVIRONMENT_H_

#include "config.hpp"
#include "global.hpp"
#include "food_source.hpp"
#include "time.hpp"
#include "distance.hpp"
#include "age.hpp"
#include "energy.hpp"

#include <blitz/array.h>
#include <memory>
using namespace blitz;


class Environment {
public:
  Environment (const Config *);
  ~Environment ();

  Array<double,1> dot_state (const Array<double,1>,
                             const Array<double,1>);
  
  void add_food_source (ptr_food_source);
  void remove_food_source (int); // index is zero based!

  bool state_is_final (const Array<double,1>);
  double get_final_reward (const Array<double,1>);
  Array<double,1> dd_state_final_reward (const Array<double,1>);
  double get_energy (const Array<double,1>,
                     const Array<double,1>);
  double dist_to_home (const Array<double,1>);

  Array<double,1> dd_action_energy (const Array<double,1>);
  Array<double,2> dd_action_dot_state (const Array<double,1>);
  Array<double,2> dd_state_dot_state (const Array<double,1>);
  Array<double,1> dd_state_energy (const Array<double,1>);

  std::vector<Array<double,1> > food_sources_locations ();

  bool test (const Array<double,1>,
             const Array<double,1>);

private:
  const Config * CONFIG;
  
  Distance * distance;
  Time * time;
  Age * age;
  Energy * energy;
  FoodSources food_sources;
  
};


typedef std::shared_ptr<Environment> ptr_environment;

#endif
