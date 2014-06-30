#ifndef _LEARNING_H_
#define _LEARNING_H_


#include "config.hpp"
#include "global.hpp"

#include <memory>


class ILearning {
public:
virtual ~ILearning(){;};

virtual std::pair<double,double> learn(const  Trajectory &,
                                         const std::shared_ptr<NeuralNetwork>,
                                         const std::shared_ptr<NeuralNetwork>)=0;


virtual bool test(std::shared_ptr<NeuralNetwork>,
                    std::shared_ptr<NeuralNetwork>)=0;

private:
ILearning& operator=(const ILearning&);

};




#endif
