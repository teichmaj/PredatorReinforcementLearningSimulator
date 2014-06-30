#ifndef PREDATOR_H_
#define PREDATOR_H_


#include <blitz/array.h>
#include <chrono>
#include <list>

#include "global.hpp"

#include "config.hpp"
#include "NN/neural_network.hpp"
#include "learning.hpp"
#include "environment.hpp"

using namespace blitz;


class Predator {
public:
  Predator (const Config *CONFIG_,
            ptr_environment,
            bool,
            int _learn_model = 1);

  ~Predator ();

  void run_episode(int);
  Trajectory get_trajectory();
  void set_Actor();
  void set_Actor(const std::vector<int> &);
  void set_Critic();
  void set_Critic(const std::vector<int> &);
  void load ();
  void save ();
  void save (char const *);
  

private:

  int learn_model;
  ILearning * learning;
  std::shared_ptr<Environment> environment;

  double _string_to_double (const std::string &);
  void _output_trajectory (const Trajectory &);
  void _output_state ();
  void _output_energy ();
  Array<double,1> get_action();

  const Config * CONFIG;
  Array<double,1> state;
  Array<double,1> action;  
  
  std::vector<double> ENERGY;
  
  std::shared_ptr<NeuralNetwork> Actor;
  std::shared_ptr<NeuralNetwork> Critic;

};


#endif
