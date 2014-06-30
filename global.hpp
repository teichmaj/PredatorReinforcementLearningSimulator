#ifndef _GLOBAL_H_
#define _GLOBAL_H_

#include <blitz/array.h>
#include <vector>
using namespace blitz;


static const int TIME = 0;
static const int AGE = 1;
static const int D1 = 2;
static const int D2 = 3;

static const int dE1 = 0;
static const int dE2 = 1;


typedef std::pair<Array<double,1>, Array<double,1> > State_action_pair;
typedef std::vector<State_action_pair> Trajectory;



#endif
