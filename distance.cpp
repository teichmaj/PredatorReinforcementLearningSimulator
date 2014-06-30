#include "distance.hpp"
#include "config.hpp"
#include "global.hpp"

#include <blitz/array.h>
#include <csignal>

using namespace blitz;


// ***********************

Distance::Distance (const Config *CONFIG_)
{
  CONFIG = CONFIG_;
}

Distance::~Distance ()
{
  ;
}

Array<double,1> Distance::dot_distance(const Array<double,1> state,
                                       const Array<double,1> action)
{
  Array<double,1> a(2);
  a = _locomotion(action);
 
  Array<double,1> result_dot_distance(4);
  result_dot_distance = 0;
  result_dot_distance(D1) = a(dE1);
  result_dot_distance(D2) = a(dE2);
  return result_dot_distance;
}

Array<double,2> Distance::dd_state(const Array<double,1> state)
{
  Array<double,2> result_dd_state(4,4);
  result_dd_state = 0;
  return result_dd_state;
}

Array<double,2> Distance::dd_action(const Array<double,1> action)
{
  Array<double,1> dd(2);
  dd = _dd_locomotion(action);
 
  Array<double,2> result_dd_action(4,2);
  result_dd_action = 0;
  result_dd_action(D1,dE1) = dd(0);
  result_dd_action(D2,dE2) = dd(1);
  return result_dd_action;
}

Array<double,1> Distance::_locomotion(const Array<double,1> action)
{
  Array<double,1> result_locomotion(2);
  result_locomotion = tanh(CONFIG->d_0 * action);
  return result_locomotion;
}


Array<double,1> Distance::_dd_locomotion(const Array<double,1> action)
{
  Array<double,1> a(2);
  a = _locomotion(action);
  Array<double,1> result_dd_locomotion(2);
  result_dd_locomotion = CONFIG->d_0 * (1.0 - (a*a));
  return result_dd_locomotion;
}


double Distance::dist_to_home(const Array<double,1> state)
{
  double result_dist_to_home;
  result_dist_to_home = sqrt(pow(state(D1),2.0) + pow(state(D2),2.0));
  return result_dist_to_home;
}

Array<double,1> Distance::dd_state_dist_to_home(const Array<double,1> state)
{
  Array<double,1> result_dd_state_dist_to_home(4);
  result_dd_state_dist_to_home = 0;

  double dist = dist_to_home(state);

  result_dd_state_dist_to_home(D1) = state(D1) / dist;
  result_dd_state_dist_to_home(D2) = state(D2) / dist;

  return result_dd_state_dist_to_home;
}


bool Distance::test(const Array<double,1> state,
		    const Array<double,1> action)
{
  double h = 0.0001;
  bool all_pass = true;
  
  // state derivative
  //
  Array<double,1> _hs(4);
  Array<double,2> res_st(4,4);
  Array<double,1> res_dist_home(4);

  for (int i=0; i<4; i++)
    {
      _hs = 0;
      _hs(i) = h;
      Array<double,1> s1(4);
      s1 = state + _hs;
      Array<double,1> s2(4);
      s2 = state - _hs;
      res_st(Range::all(),i) = (dot_distance(s1, action) -	\
		   dot_distance(s2, action)) / (2*h);
      res_dist_home(i) = (dist_to_home(s1) - dist_to_home(s2)) / (2*h);
    }

  Array<double,2> dState (4,4);
  dState = dd_state(state);
   if ( sum(abs(res_st-dState)) > 0.001) {
     all_pass = false;
     std::cout << "Distance::test() state derivative failed!\n";
     std::cout << "derivative: " << dState << std::endl;
     std::cout << "stencil: " << res_st << std::endl;
   }
   Array<double,1> dDistHome(4);
   dDistHome = dd_state_dist_to_home(state);
   if ( sum(abs(res_dist_home-dDistHome)) > 0.001) {
     all_pass = false;
     std::cout << "Distance::test() state derivative of dist_to_home failed!\n";
     std::cout << "derivative: " << dDistHome << std::endl;
     std::cout << "stencil: " << res_dist_home << std::endl;
   }



  // action derivative
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
      res_ac(Range::all(),i) = (dot_distance(state, a1) -	\
		   dot_distance(state, a2)) / (2*h);
    }

  Array<double,2> dAction (4,2);
  dAction = dd_action(action);
   if ( sum(abs(res_ac-dAction)) > 0.001) {
     all_pass = false;
     std::cout << "Distance::test() action derivative failed!\n";
     std::cout << "derivative: " << dAction << std::endl;
     std::cout << "stencil: " << res_ac << std::endl;
   }
  
   return all_pass;
}











