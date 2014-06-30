
#include "config.hpp"
#include "gui.hpp"
#include "predator.hpp"

#include <blitz/array.h>
#include <signal.h>
#include <stdlib.h>
#include <unistd.h>

using namespace blitz;

sig_atomic_t stopFlag = 0;

void stop_handler(int s){
  stopFlag = 1;
}

int main(int argc, char* argv[])
{
  Config * CONFIG = new Config();
  CONFIG->use_defaults();

  srand(CONFIG->rnd_seed);

  // Adding a food source to the environment
  Array<double,1> mu(2);
  mu = 5;
  Array<double,1> sig(2);
  sig = 5;
  Gaussian gaus(mu, sig);
  ptr_food_source food_source(new FoodSource<Gaussian> (CONFIG,
                                                         gaus,
							 2.0, // tox
                                                         2.0, // rew
                                                         0.1, // time
                                                         0.5)); // p
  ptr_environment environment(new Environment(CONFIG));
  environment->add_food_source(food_source);

  // ceating a predator
  Predator predator(CONFIG, environment, false, 2);
  std::vector<int> layout_actor;
  layout_actor = {4,50,2};
  predator.set_Actor(layout_actor);
  std::vector<int> layout_critic;
  layout_critic = {4,50,4};
  predator.set_Critic(layout_critic);
  
  struct sigaction sigIntHandler;
  sigIntHandler.sa_handler = stop_handler;
  sigemptyset(&sigIntHandler.sa_mask);
  sigIntHandler.sa_flags = 0;
  sigaction(SIGINT, &sigIntHandler, NULL);
  
  int i = 1;
  while (!stopFlag) 
    {    
      predator->run_episode(i);
      i ++;
    }

  predator->save();
  
  delete CONFIG;
  delete predator;

  return 1;
}
