#include "time.hpp"
#include "config.hpp"
#include "food_source.hpp"
#include "global.hpp"

#include <blitz/array.h>
#include <csignal>

using namespace blitz;


Time::Time (const Config *CONFIG_)
{
  CONFIG = CONFIG_;
}

Time::~Time ()
{
  ;
}

Array<double,1> Time::dot_time(const Array<double,1> state,
                               const FoodSources food_sources)
{
  Array<double,1> result(4);
  result = 0;
  result(TIME) = 1.0;

  FoodSources::const_iterator food_it;
  for (food_it = food_sources.begin();
       food_it != food_sources.end();
       ++food_it)
    {
      result(TIME) += (*food_it)->get_time(state);
    }

  return result;
}


Array<double,2> Time::dd_state (const Array<double,1> state,
                          const FoodSources food_sources)
{
  Array<double,2> result(4,4);
  result = 0;

  FoodSources::const_iterator food_it;
  for (food_it = food_sources.begin();
       food_it != food_sources.end();
       ++food_it)
    {
      result += (*food_it)->dd_T_state(state);
    }
  
  return result;
}

Array<double,2> Time::dd_action(const Array<double,1> action,
                                const FoodSources food_sources)
{
  Array<double,2> result(4,2);
  result = 0;
  FoodSources::const_iterator food_it;
  for (food_it = food_sources.begin();
       food_it != food_sources.end();
       ++food_it)
    {
      result += (*food_it)->dd_T_action(action);
    }

  return result;
}

bool Time::test (const Array<double,1> state,
                 const Array<double,1> action,
                 const FoodSources food_sources)
{
  double h = 0.0001;
  bool all_pass = true;

  // state derivative
  //
  Array<double,1> _hs(4);
  Array<double,2> res_st(4,4);
  
  for (int i=0; i<4; i++)
    {
      _hs = 0;
      _hs(i) = h;
      Array<double,1> s1(4);
      s1 = state + _hs;
      Array<double,1> s2(4);
      s2 = state - _hs;
      res_st(Range::all(),i) = (dot_time(s1, food_sources) - \
                                dot_time(s2, food_sources) ) / (2*h);
    }

  Array<double,2> dState (4,4);
  dState = dd_state(state, food_sources);
   if ( sum(abs(res_st-dState)) > 0.001) {
     all_pass = false;
     std::cout << "Time::test() state derivative failed!\n";
   }
   
   return all_pass;
}
