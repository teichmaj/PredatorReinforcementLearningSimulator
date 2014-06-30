#ifndef _REWARD_H_
#define _REWARD_H_

#include "global.hpp"
#include "config.hpp"
#include "food_source.hpp"


class Reward {
public:
  Reward (const Config *);
  ~Reward ();

  double get_reward (const Array<double,1>,
                     const FoodSources);

  Array<double,1> dd_state (const Array<double,1>,
                            const FoodSources);
  Array<double,1> dd_action (const Array<double,1>,
                             const FoodSources);
  
  bool test (const Array<double,1>,
             const Array<double,1>,
             const FoodSources);
  
private:
  const Config * CONFIG;
};

#endif
  
