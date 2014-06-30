#include "age.hpp"

#include "config.hpp"
#include "global.hpp"
#include "time.hpp"

#include <blitz/array.h>
using namespace blitz;


Age::Age (const Config *CONFIG_)
{
  CONFIG = CONFIG_;
}

Age::~Age ()
{
  ;
}

double Age::age_agility(const Array<double,1> state)
{
  double result;
  result = 1.0 / (1.0 + state(AGE));
  return result;
}

double Age::dd_age_agility(const Array<double,1> state)
{
  double result;
  result = - 1.0 / 
    pow(state(AGE)+1.0,2.0);
  return result;
}

Array<double,1> Age::dot_age(const Array<double,1> state,
                             const FoodSources food_sources)
{
  Array<double,1> result(4);
  result = 0;

  Time time(CONFIG);
  double t = time.dot_time(state, food_sources)(TIME);
  
  result(AGE) = (1.0/CONFIG->lambda_0) * t;
  return result;
}

Array<double,2> Age::dd_action (const Array<double,1> action,
                                const FoodSources food_sources)
{
  Array<double,2> result(4,2);
  result = 0;
  
  Time time(CONFIG);
  result = 1.0/CONFIG->lambda_0 * time.dd_action(action,food_sources);
  
  return result;
}

Array<double,2> Age::dd_state (const Array<double,1> state,
                               const FoodSources food_sources)
{
  Array<double,2> result(4,4);
  result = 0;

  Time time(CONFIG);
  Array<double,2> dt(4,4);
  dt = time.dd_state(state,food_sources);


  result(AGE,D1) = 1.0/CONFIG->lambda_0 * dt(TIME,D1);
  result(AGE,D2) = 1.0/CONFIG->lambda_0 * dt(TIME,D2);

  return result;
}

bool Age::test (const Array<double,1> state,
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
      res_st(Range::all(),i) = (dot_age(s1,food_sources) - 
                                dot_age(s2,food_sources)) / (2.0*h); 
    }

  Array<double,2> dState (4,4);
  dState = dd_state(state,food_sources);
   if ( sum(abs(res_st-dState)) > 0.001) {
     all_pass = false;
     std::cout << "Age::test() state derivative failed!\n";
   }
   
   double dd_agility;
   _hs = 0;
   _hs(AGE) = h;
   Array<double,1> s1(4);
   s1 = state + _hs;
   Array<double,1> s2(4);
   s2 = state - _hs;
   dd_agility = (age_agility(s1) - age_agility(s2)) / (2*h);
   
   double dAgility = dd_age_agility(state);

   if ( abs(dd_agility-dAgility) > 0.001) {
     all_pass = false;
     std::cout << "Age::test() derivative age_agility failed!\n";
     std::cout << "stencil: " << dd_agility << std::endl;
     std::cout << "derivative: " << dAgility << std::endl;
   }


   // action derivative
   // 
   Array<double,2> dAction (4,2);
   dAction = dd_action(action, food_sources);
   if ( sum(abs(dAction)) > 0.001) {
     all_pass = false;
     std::cout << "Age::test() action derivative failed!\n";
   }
  

   return all_pass;
}
