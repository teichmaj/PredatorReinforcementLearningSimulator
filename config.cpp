#include "config.hpp"
#include <string>
#include <vector>
#include <sstream>
#include <iostream>
#include <fstream>
#include <ios>
#include <ctime>


Config::Config () {}

Config::~Config () {}



std::vector<std::string> Config::split(std::string str, std::string delim) const
{ 
      int start = 0;
      int end; 
      std::vector<std::string> v; 

      while( (end = str.find(delim, start)) != std::string::npos )
      { 
            v.push_back(str.substr(start, end-start)); 
            start = end + delim.length(); 
      } 
      v.push_back(str.substr(start)); 
      return v; 
}

int Config::_as_int(std::string s) const {
     int result;
     std::stringstream convert(s); 

     if ( !(convert >> result) ) {
          result = 0;
     }
     return result;
}
    
double Config::_as_double (std::string s) const {
     double result;
     std::stringstream convert(s);

      if ( !(convert >> result) ) {
          result = 0;
     }
     return result;
}
    
     

void Config::load(char *fn)
{
     std::ifstream infile;
     infile.open(fn);
     if (infile.is_open()) {
          std::string line;
          bool conf_flag = false;
          std::vector<std::string> v;
           while ( infile.good() )
                {
                     std::getline(infile, line);
                     if (line == "###CONFIG") {
                          conf_flag = true;
                          continue;
                     }                          
                     v = split(line,"=");
                     
                     if (!conf_flag) {
                          continue;
                     } else {
                     
                          if (v[0] == "rnd_seed"){
                               rnd_seed = _as_int(v[1]);
                          } else if (v[0] == "time_sampling"){
                               time_sampling = _as_double(v[1]);
                          } else if (v[0] == "time_toxin"){
                               time_toxin = _as_double(v[1]);
                          } else if (v[0] == "home_dist_punishment"){
                               home_dist_punishment = _as_double(v[1]);
                          } else if (v[0] == "home_dist_tolerance"){
                               home_dist_tolerance = _as_double(v[1]);
                          } else if (v[0] == "s_0"){
                               s_0 = _as_double(v[1]);
                          } else if (v[0] == "d_0"){
                               d_0 = _as_double(v[1]);
                          } else if (v[0] == "t_0"){
                               t_0 = _as_double(v[1]);
                          } else if (v[0] == "lambda_0"){
                               lambda_0 = _as_double(v[1]);
                          } else if (v[0] == "length_of_episode"){
                               length_of_episode = _as_double(v[1]);
                          } else if (v[0] == "alpha"){
                               alpha = _as_double(v[1]);
                          } else if (v[0] == "gamma"){
                               gamma = _as_double(v[1]);
                          } else if (v[0] == "nn_layout"){
                               std::vector<std::string> tmp;
                               tmp = split(v[1],"{");
                               std::vector<std::string> n;
                               n = split(tmp[1],",");
                               std::vector<std::string>::iterator i;
                               for (i=n.begin(); i<n.end(); ++i) {
                                    nn_layout.push_back(_as_int(*i));
                               }
                          } else {
                               std::cout << "unknown: " << line << std::endl;
                          } 
                     }
                }
     }
     infile.close();
}


void Config::save(char const *fn) const
{
     std::ofstream outfile;
     outfile.open(fn, std::ios::out | std::ios::app);
     if (outfile.is_open()) {
          outfile << "###CONFIG\n";
          outfile << "rnd_seed=" << rnd_seed << std::endl;
          outfile << "time_sampling=" << time_sampling << std::endl;
          outfile << "time_toxin=" << time_toxin << std::endl;
          outfile << "home_dist_punishment=" << home_dist_punishment << std::endl;
          outfile << "home_dist_tolerance=" << home_dist_tolerance << std::endl;
          outfile << "s_0=" << s_0 << std::endl;
          outfile << "d_0=" << d_0 << std::endl;
          outfile << "t_0=" << t_0 << std::endl;          
          outfile << "lambda_0=" << lambda_0 << std::endl;
          outfile << "length_of_episode=" << length_of_episode << std::endl;
          outfile << "alpha=" << alpha << std::endl;
          outfile << "gamma=" << gamma << std::endl;
          outfile << "nn_layout={";
          std::vector<int>::const_iterator i;
          for (i = nn_layout.begin(); i<nn_layout.end(); ++i) {
               outfile << *i;
               if (i< --nn_layout.end()) {
                    outfile << ",";
               }
          }
          outfile << "}\n";
     }
     outfile.close();
}

void Config::use_defaults ()
{
  rnd_seed = 1;

  time_sampling = 0.1;
  time_toxin = 0.1;

  home_dist_punishment = -10;
  home_dist_tolerance = 0.1;

  s_0 = 0.1;
  d_0 = 1;
  t_0 = 1;
  lambda_0 = 1000;
  
  length_of_episode = 25;

  alpha = 0.1; // learning rate
  gamma = 0.9; // discount rate for future rewards

  nn_layout = {4,10,2};

  std::time_t t = std::time(0);
  t_stamp = std::to_string(t);
}
