#include "reward.hpp"
#include "config.hpp"
#include "global.hpp"

#include <blitz/array.h>
using namespace blitz;



Reward::Reward (const Config *CONFIG_)
{
  CONFIG = CONFIG_;
}

Reward::~Reward ()
{
  ;
}

double Reward::get_reward (const Array<double,1> state,
                           const FoodSources food_sources)
{
  double reward = 0;
  FoodSources::const_iterator food_it;
  for (food_it = food_sources.begin();
       food_it != food_sources.end();
       ++food_it)
    {
      reward += (*food_it)->get_reward(state);
    }
  return reward;
}


Array<double,1> Reward::dd_action (const Array<double,1> action,
                                   const FoodSources food_sources)
{
  Array<double,1> result(2);
  result = 0;
  
  FoodSources::const_iterator food_it;
  for (food_it = food_sources.begin();
       food_it != food_sources.end();
       ++food_it)
    {
      result += (*food_it)->dd_R_action(action);
    }

  return result;
}


Array<double,1> Reward::dd_state (const Array<double,1> state,
                                  const FoodSources food_sources)
{
  Array<double,1> result(4);
  result = 0;
  
  FoodSources::const_iterator food_it;
  for (food_it = food_sources.begin();
       food_it != food_sources.end();
       ++food_it)
    {
      result += (*food_it)->dd_R_state(state);
    }

  return result;
}


bool Reward::test(const Array<double,1> state,
                  const Array<double,1> action,
                  const FoodSources food_sources)
{
  double h = 0.0001;
  bool all_pass = true;
  
  // state derivative REWARD
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
      res_st(i) = (get_reward(s1, food_sources)- 
                   get_reward(s2, food_sources)) / (2*h);
    }
		
  Array<double,1> dState (4);
  dState = dd_state(state, food_sources);
  if ( sum(abs(res_st-dState)) > 0.001) {
    all_pass = false;
    std::cout << "Reward::test() state derivative failed!\n";
    std::cout << "derivative: " << dState << std::endl;
    std::cout << "stencil: " << res_st << std::endl;
  }


  return all_pass;
}
