#ifndef CONFIG_H_
#define CONFIG_H_

#include <vector>
#include <string>


class Config {
public:
  Config ();
  ~Config ();
  void use_defaults ();
  void load(char *);
  void save(char const *) const;

  int rnd_seed;             // random seed  
  double time_sampling;
  double time_toxin;
  double home_dist_punishment;  // punishment for not being home at 
                                // end of episode
  double home_dist_tolerance;   // tolerance for home_dist_punishment
  double s_0;                // taste sampling efficiency
  double d_0;                // distance scaling 
  double t_0;             // metabolic rate
  double lambda_0;           // speed of agility decline
  double lambda_vgl;         // discount factor VGL
  double length_of_episode;  // length of day
  std::vector<int> nn_layout;     // NN layout
  double alpha;              // learning rate
  double gamma;              // discount factor for future rewards

  std::string t_stamp;

private:
     std::vector<std::string> split(std::string,std::string) const;
     int _as_int(std::string) const;
     double _as_double(std::string) const;
};




#endif
