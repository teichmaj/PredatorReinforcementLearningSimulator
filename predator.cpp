#include "predator.hpp"
#include "global.hpp"

#include "NN/neural_network.hpp"
#include "config.hpp"
#include "bptt.hpp"
#include "vgl.hpp"
#include "learning.hpp"
#include "environment.hpp"

#include <blitz/array.h>
#include <iostream>
#include <fstream>
#include <string>
#include <boost/foreach.hpp>
#include <chrono>
#include <list>
#include <vector>
#include <numeric>
#include <csignal>

using namespace blitz;



// ========================================
// public

Predator::Predator (const Config *CONFIG_,
                    const std::shared_ptr<Environment> env_,
                    const bool is_rprop,
                    const int _learn_model)
{
  // learn_model = 1: BPTT
  // learn_model = 2: VGL
  
  learn_model = _learn_model;
  CONFIG = CONFIG_;
  state.resize(4);
  state = 0;
  
  action.resize(2);
  action = 0;
  
  environment = env_;
  if (learn_model == 1) {
    learning = new BPTT(CONFIG, environment, is_rprop);
  }
  else if (learn_model == 2)
    {
      learning = new VGL(CONFIG, environment, is_rprop);
    }

}

Predator::~Predator ()
{
    delete learning;
}

Trajectory Predator::get_trajectory()
{
  Trajectory trajectory;
  ENERGY.clear();
  ENERGY.push_back(0);
   
  while (!environment->state_is_final(state)) {
    action = get_action();
    trajectory.push_back(std::make_pair(state.copy(), action.copy()));
    ENERGY.push_back(environment->get_energy(state, action));
    Array<double,1> _dot_state(4);
    _dot_state = environment->dot_state(state, action);
    state += _dot_state;
  }

  action = 0;
  trajectory.push_back(std::make_pair(state, action));
  ENERGY.push_back(environment->get_final_reward(state));

  return trajectory;
}


Array<double,1> Predator::get_action()
{
  Array<double,1> _a(2);
  _a = Actor->inputResponse(state);
  return _a;
}

void Predator::run_episode(int round)
{
  bool m = false;
  state = 0;
  state(AGE) = (0.1 * rand() / (RAND_MAX + 1.0));
  state(D1) = (0.2 * rand() / (RAND_MAX + 1.0));
  state(D2) = (0.2 * rand() / (RAND_MAX + 1.0));
  state(TIME) = (0.2 * rand() / (RAND_MAX + 1.0));
  
  
  Trajectory trajectory;
  trajectory = get_trajectory();

  std::pair<double,double> delta = {0,0};
    
  delta = learning->learn(trajectory, Actor, Critic);  

  double total_energy = std::accumulate(ENERGY.begin(), 
                                        ENERGY.end(), 0);

  bool is_home;
  is_home = environment->dist_to_home(state) < CONFIG->home_dist_tolerance;

}

void Predator::save ()
{
  delete gui;
  char fn[20];
  cout << "\n\nFile name [max len 20] to save NN: ";
  cin >> fn;
  save(fn);
  std::cout << "NN saved successfully...\n";
  gui = new Gui (CONFIG);
}

void Predator::save(char const *fn) {
  std::vector<double> weights;
  weights = Actor->get_weights_flattened();
  ofstream out_file (fn);
  if (out_file.is_open()) {
    std::vector<double>::iterator i;
    out_file << "###SCORE: " << max_energy << "\n";
    out_file << "###NN_Actor\n";
    for (i = weights.begin(); i < weights.end(); ++i) {
      out_file << *i;
      out_file << endl;
    }

    if (learn_model == 2) {
      out_file << "###NN_Critic\n";
      weights = Critic->get_weights_flattened();
      for (i = weights.begin(); i < weights.end(); ++i) {
        out_file << *i;
        out_file << endl;
      }
    }

    out_file.close();
    CONFIG->save(fn);
  }
}

void Predator::load ()
{
  char fn[20];
  cout << "\n\nload NN file name : ";
  cin >> fn;
  
  ifstream in_file (fn);
  if (in_file.is_open()) {
    string line;
    std::vector<double> weights;
    bool nn_flag = false;

    while ( in_file.good() )
         {
              std::getline(in_file, line);
              if (line == "###NN") {
                   nn_flag = true;
                   continue;
              } else if (line == "###CONFIG") {
                   nn_flag = false;
                   continue;
              }
              if (!nn_flag) {
                   continue;
              } else {
              
                   weights.push_back(_string_to_double(line));
              }
         }
    Actor->set_weights(weights);
    std::cout << "\nsuccessfully loaded NN from file\n";
  }
}

// ======================================

double Predator::_string_to_double( const std::string& s )
{
   std::istringstream i(s);
   double x;
   if (!(i >> x)) {
     return 0;
   }
   return x;
} 

void Predator::_output_state () {
  std::cout << "\nFinal State\n====================\n\n";
  std::cout << state;
}

void Predator::_output_energy ()
{
  std::cout << "\nENERGY\n========================\n\n";
  std::cout << ENERGY.back() << "\n";

}

void Predator::_output_trajectory (const Trajectory &trajectory)
{
  int n = 0;
  std::cout << "\nTrajectory:\n=========================\n\n";
  BOOST_FOREACH(const State_action_pair &pair, trajectory) {
    std::cout << "["<<n<<"] \nState\n    ";
    std::cout << pair.first;
    std::cout << "Action\n    ";
    std::cout << pair.second << "\n";
    n ++;
  }
  
}

void Predator::set_Actor ()
{
     NeuralNetwork net(CONFIG->nn_layout);
     Actor = std::make_shared<NeuralNetwork> (net); 
}

void Predator::set_Actor(const std::vector<int> &layout)
{
     NeuralNetwork net(layout);
     Actor = std::make_shared<NeuralNetwork> (net);  
}

void Predator::set_Critic ()
{
     NeuralNetwork net(CONFIG->nn_layout);
     Critic = std::make_shared<NeuralNetwork> (net); 
}

void Predator::set_Critic(const std::vector<int> &layout)
{
     NeuralNetwork net(layout);
     Critic = std::make_shared<NeuralNetwork> (net);  
}
