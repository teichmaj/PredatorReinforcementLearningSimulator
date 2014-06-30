#include "environment.hpp"
#include "config.hpp"
#include "global.hpp"
#include "distance.hpp"
#include "time.hpp"
#include "energy.hpp"
#include "reward.hpp"

#include <blitz/array.h>
#include <cassert>
#include <boost/foreach.hpp>
#include <csignal>

#include <iostream>
#include <fstream>

using namespace blitz;


Environment::Environment (const Config * CONFIG_)
{
  CONFIG = CONFIG_;
  distance = new Distance (CONFIG);
  time = new Time (CONFIG);
  age = new Age (CONFIG);
  energy = new Energy (CONFIG);
}

Environment::~Environment ()
{
  delete distance;
  delete time;
  delete age;
  delete energy;
}


Array<double,1> Environment::dot_state (const Array<double,1> state,
                                        const Array<double,1> action)
{
  Array<double,1> result(4);
  result = 0;

  result += distance->dot_distance(state, action);
  result += time->dot_time(state, food_sources);
  result += age->dot_age(state, food_sources);

  return result;
}

Array<double,2> Environment::dd_action_dot_state(const Array<double,1> action)
{
  Array<double,2> result(4,2);
  result = 0;
  result += distance->dd_action(action);
  result += time->dd_action(action,food_sources);
  result += age->dd_action(action,food_sources);

  return result;
}

Array<double,2> Environment::dd_state_dot_state(const Array<double,1> state)
{
  Array<double,2> result(4,4);
  result = 0;

  result += distance->dd_state(state);
  result += time->dd_state(state, food_sources);
  result += age->dd_state(state, food_sources);
  
  return result;
}

double Environment::dist_to_home (const Array<double,1> state)
{
  return distance->dist_to_home(state);
}

void Environment::add_food_source (ptr_food_source food_source)
{
  food_sources.push_back(food_source);
}

void Environment::remove_food_source (int idx)
{
  // index is zero based!
  food_sources.erase(food_sources.begin()+idx);
}

bool Environment::state_is_final (const Array<double,1> state)
{
   return state(TIME) > CONFIG->length_of_episode;
}

double Environment::get_energy (const Array<double,1> state,
                                const Array<double,1> action)
{
  double e;
  e = energy->dot_energy(state, action, food_sources);
  return e;
}

double Environment::get_final_reward(const Array<double,1> state)
{
  assert (state_is_final(state));
  double dist = distance->dist_to_home(state);
  double result = 0;
  if (dist > CONFIG->home_dist_tolerance) {
    // result = CONFIG->home_dist_punishment *   \
    // (dist - CONFIG->home_dist_tolerance);
    result = CONFIG->home_dist_punishment * (dist-CONFIG->home_dist_tolerance);
  }
  return result;
}

std::vector<Array<double,1> > Environment::food_sources_locations ()
{
  std::vector<Array<double,1> > result;
  BOOST_FOREACH(const ptr_food_source &food_source, food_sources)
    {
      result.push_back(food_source->location());
    }

  return result;
}

Array<double,1> Environment::dd_state_final_reward (const Array<double,1> state)
{
  assert (state_is_final(state));
  double dist = distance->dist_to_home(state);

  Array<double,1> result(4);
  result = 0;

  if (dist > CONFIG->home_dist_punishment) {
    result = distance->dd_state_dist_to_home(state) *   \
      CONFIG->home_dist_punishment;
  }

  return result;
}

Array<double,1> Environment::dd_action_energy(const Array<double,1> action)
{
  Array<double,1> result(2);
  result = energy->dd_action(action);
  return result;
}


Array<double,1> Environment::dd_state_energy(const Array<double,1> state)
{
  Array<double,1> result(4);
  result = energy->dd_state(state, food_sources);
  return result;
}


bool Environment::test(const Array<double,1> state,
                       const Array<double,1> action)
{
  double h = 0.0001;
  bool all_pass = true;

  // state derivative
  //
  Array<double,1> _hs(4);
  Array<double,1> res_final_rew(4);
  Array<double,1> state_f(state);
  state_f(TIME) = CONFIG->length_of_episode + 1;
  Array<double,2> res_st(4,4);
  Array<double,1> res_en(4);

   for (int i=0; i<4; i++)
    {
      _hs = 0;
      _hs(i) = h;
      Array<double,1> s1(4);
      s1 = state_f + _hs;
      Array<double,1> s2(4);
      s2 = state_f - _hs;
      res_final_rew(i) = (get_final_reward(s1)-get_final_reward(s2)) / (2*h);

      s1 = state + _hs;
      s2 = state - _hs;
      res_st(Range::all(),i) = (dot_state(s1,action) - dot_state(s2,action)) / (2*h);

      res_en(i) = (get_energy(s1,action) - get_energy(s2,action)) / (2*h);
    }

   Array<double,1> dFinalRew(4);
   dFinalRew = dd_state_final_reward(state_f);
   if ( sum(abs(res_final_rew-dFinalRew)) > 0.001) {
     all_pass = false;
     std::cout << "Environment::test() state derivative of get_final_reward failed!\n";
     std::cout << "stencil:" << res_final_rew << std::endl;
     std::cout << "derivative:" << dFinalRew << std::endl;
   }

    Array<double,2> dDotState(4,4);
    dDotState = dd_state_dot_state(state);
    if ( sum(abs(res_st-dDotState)) > 0.001) {
     all_pass = false;
     std::cout << "Environment::test() state derivative of dot_state failed!\n";
     std::cout << "stencil:" << res_st << std::endl;
     std::cout << "derivative:" << dDotState << std::endl;
    }
    
    Array<double,1> dEn(4);
    dEn = dd_state_energy(state);
    if ( sum(abs(res_en-dEn)) > 0.0001) {
     all_pass = false;
     std::cout << "Environment::test() state derivative of get_energy failed!\n";
     std::cout << "stencil:" << res_en << std::endl;
     std::cout << "derivative:" << dEn << std::endl;
    }
    

    // action derivatives
    // 
    Array<double,1> _ha(2);
  Array<double,2> res_ac(4,2);
  
  for (int i=0; i<2; i++)
    {
      _ha = 0;
      _ha(i) = h;
      Array<double,1> a1(2);
      a1 = action + _ha;
      Array<double,1> a2(2);
      a2 = action - _ha;
      res_ac(Range::all(),i) = (dot_state(state,a1) - dot_state(state,a2)) / (2*h);
    }
  
  Array<double,2> dA(4,2);
  dA = dd_action_dot_state(action);
  if ( sum(abs(res_ac-dA)) > 0.001) {
     all_pass = false;
     std::cout << "Environment::test() action derivative failed!\n";
     std::cout << "stencil:" << res_ac << std::endl;
     std::cout << "derivative:" << dA << std::endl;
  }

  return all_pass;
}


