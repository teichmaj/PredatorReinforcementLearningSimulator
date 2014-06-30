#include "energy.hpp"
#include "config.hpp"
#include "global.hpp"
#include "sgn.hpp"

#include "age.hpp"
#include "reward.hpp"
#include "time.hpp"

#include <blitz/array.h>
#include <cmath>
#include <csignal>
using namespace blitz;


Energy::Energy (const Config *CONFIG_)
{
  CONFIG = CONFIG_;
}

Energy::~Energy ()
{
  ;
}

double Energy::dot_energy (const Array<double,1> state,
                     const Array<double,1> action,
                     const FoodSources food_sources)
{
  double result = 0;

  Age age(CONFIG);
  Time time(CONFIG);
  Reward reward(CONFIG);
  
  result = age.age_agility(state)*reward.get_reward(state, food_sources);
  result -= CONFIG->t_0 * time.dot_time(state,food_sources)(TIME);
  result -= std::abs(action(dE1));
  result -= std::abs(action(dE2));

  return result;
}

Array<double,1> Energy::dd_action (const Array<double,1> action)
{
  Array<double,1> result(2);
  result = 0;
  
  result(dE1) = -sgn<double>(action(dE1));
  result(dE2) = -sgn<double>(action(dE2));
 
  return result;
}


Array<double,1> Energy::dd_state (const Array<double,1> state,
                                    const FoodSources food_sources)
{
  Array<double,1> result(4);
  
  Age age(CONFIG);
  Reward reward(CONFIG);
  Time time(CONFIG);
  
  result(TIME) = 0;
  
  result(AGE) = age.dd_age_agility(state) * reward.get_reward(state, food_sources);
  
  result(D1) = age.age_agility(state) * reward.dd_state(state,food_sources)(D1) - \
       (CONFIG->t_0 * time.dd_state(state,food_sources)(TIME,D1));
  
  result(D2) = age.age_agility(state) * reward.dd_state(state,food_sources)(D2) - \
       (CONFIG->t_0 * time.dd_state(state,food_sources)(TIME,D2));
  
  return result;
}



bool Energy::test (const Array<double,1> state,
                     const Array<double,1> action,
                     const FoodSources food_sources)
{
  double h = 0.0001;
  bool all_pass = true;

  // state derivative
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
      res_st(i) = (dot_energy(s1,action,food_sources) - 
		   dot_energy(s2,action,food_sources)) / (2.0*h); 
    }

  Array<double,1> dState (4);
  dState = dd_state(state,food_sources);
  if ( sum(abs(res_st-dState)) > 0.001) {
    all_pass = false;
    std::cout << "Energy::test() state derivative failed!\n";
    std::cout << "stencil:" << res_st << std::endl;
    std::cout << "derivative:" << dState << std::endl;
  }
   

  // action derivative
  //
  Array<double,1> _ha(2);
  Array<double,1> res_ac(2);
  
  for (int i=0; i<2; i++)
    {
      _ha = 0;
      _ha(i) = h;
      Array<double,1> a1(2);
      a1 = action + _ha;
      Array<double,1> a2(2);
      a2 = action - _ha;
      res_ac(i) = (dot_energy(state,a1,food_sources) - 
		   dot_energy(state,a2,food_sources)) / (2.0*h); 
    }
  
  Array<double,1> dAction (2);
  dAction = dd_action(action);
  if ( sum(abs(res_ac-dAction)) > 0.001) {
    all_pass = false;
    std::cout << "Energy::test() action derivative failed!\n";
    std::cout << "stencil:" << res_ac << std::endl;
    std::cout << "derivative:" << dAction << std::endl;
  }

  return all_pass;
}



