#ifndef AGE_H_
#define AGE_H_

#include "config.hpp"
#include "global.hpp"
#include "food_source.hpp"

#include <blitz/array.h>
using namespace blitz;


class Age {
public:
Age (const Config *);
~Age ();

Array<double,1> dot_age (const Array<double,1>,
                           const FoodSources);
double age_agility (const Array<double,1>);
double dd_age_agility (const Array<double,1>);
Array<double,2> dd_action (const Array<double,1>,
                             const FoodSources);
Array<double,2> dd_state (const Array<double,1>,
                            const FoodSources);

bool test (const Array<double,1>,
             const Array<double,1>,
             const FoodSources);

private:
const Config * CONFIG;

};

#endif
