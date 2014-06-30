#ifndef _NEURALNETWORK_H_
#define _NEURALNETWORK_H_


#include <blitz/array.h>
#include <cmath>
#include <vector>
#include <memory>  // shared_ptr

using namespace blitz;


class NeuralNetwork
{
public:
  NeuralNetwork (); //empty object to receive copy
  NeuralNetwork (std::vector<int>);

  Array<double,1> inputResponse (const Array<double,1>);
  std::vector<double> get_weights_flattened ();
  void set_weights (const std::vector<double>&);

  void clone (const std::shared_ptr<NeuralNetwork>);

  Array<double,1> gradientWeights (const Array<double,1>,
			    const Array<double,1>);
   Array<double,1> gradientInput (const Array<double,1>,
			    const Array<double,1>);

  int nWeights();

private:
  std::vector<int> layout;
  std::vector<Array<double,1> > activation;
  std::vector<Array<double,2> > weight;
  
  void _init_weights_rnd ();
  void _init_weights_Nguyen_Widrow ();
  void _fwdPropagation (const Array<double,1>);
  std::vector<Array<double,1> > _deltas(const Array<double,1>);

};























#endif
