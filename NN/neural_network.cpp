#include "neural_network.hpp"


#include <blitz/array.h>
#include <numeric>
#include <cassert>
#include <vector>
#include <csignal>
#include <iostream>

using namespace blitz;



double dTanh(double x)
{
  return 1.0 - pow(x,2);
}
BZ_DECLARE_FUNCTION(dTanh)

double dOutAct(double x)
{
  // linear activation fct for output layer
  return x/x;
}
BZ_DECLARE_FUNCTION(dOutAct)

NeuralNetwork::NeuralNetwork ()
{
  layout.push_back(0);
}



NeuralNetwork::NeuralNetwork (const std::vector<int> _layout)
{
  layout = _layout;
  unsigned int j = 0;
  bool first_layer = true;
  bool last_layer = false;
  int add_bias_unit = 0;

  for (unsigned int i=0; i<layout.size(); i++)
    {	  
      if (i == (layout.size()-1))
	{
	  last_layer = true;
	}
      
      if (!first_layer && !last_layer)
	{
	  Array<double,2> w(shape(j+add_bias_unit,layout[i]));
	  // bias neuron is not connected
	  weight.push_back(w);
	  // plus one bias neuron
	  Array<double,1> a(shape(layout[i]+1));
	  // activation of bias neuron is always one
	  a(0) = 1;
	  activation.push_back(a);
	  add_bias_unit = 1;
	}
      else
	{ if (last_layer)
	    {
	      // no bias neuron in output layer
	      Array<double,1> a(shape(layout[i])); 
	      activation.push_back(a);
	      add_bias_unit = 1;
	      Array<double,2> w(shape(j+add_bias_unit,layout[i]));
	      weight.push_back(w);
	    }
	  else if (first_layer)
	    {
	      // no bias neuron in input layer
	      Array<double,1> a(shape(layout[i])); 
	      activation.push_back(a);
	      // no weights on input layer
	      first_layer = false;
	    }
	}
	  
      // number of active neurons of previous layer
      j = layout[i];
    }

  _init_weights_Nguyen_Widrow();
}

void NeuralNetwork::_init_weights_rnd ()
{
  std::vector<Array<double,2> >::iterator it;
  for (it = weight.begin() ; it != weight.end(); ++it)
    {
      TinyVector<int, 2> s = it->shape();
      for (int i=0; i<s(0); i++)
	{
	  for (int j=0; j<s(1); j++)
	    {
	      (*it)(i,j) = (0.02 * (double)rand()/(double)RAND_MAX) - 0.01;
	    }
	}
    }
}

void NeuralNetwork::_init_weights_Nguyen_Widrow ()
{
  _init_weights_rnd();
  int h = std::accumulate(layout.begin(),layout.end(),0) - \
    layout.front() - layout.back();
  double beta = 0.7 * pow(h, 1.0/layout.front());
  
  std::vector<Array<double,2> >::iterator it;
  for (it = weight.begin(); it != weight.end(); ++it)
    {
      double norm = sqrt(sum(pow((*it)(tensor::i,tensor::j),2)));
      TinyVector<int, 2> s = it->shape();
      for (int i=0; i<s(0); i++)
	{
	  for (int j=0; j<s(1); j++)
	    {
	      (*it)(i,j) = (beta * (*it)(i,j)) / norm;
	    }
	  
	}
    }
}


std::vector<double> NeuralNetwork::get_weights_flattened ()
{
  std::vector<double> ret_weights;
  
  std::vector<Array<double,2> >::iterator it;
  for (it = weight.begin() ; it != weight.end(); ++it)
    {
      /** iterator by rows over array(1st,2nd)
              ^
          2nd |
              |
              |
              |
            0 +---------->
              0         1st
      **/
      Array<double,2>::const_iterator iit;
      for (iit = it->begin(); iit != it->end(); ++iit)
	{
	  ret_weights.push_back(*iit);
	}
    }

  return ret_weights;
}


void NeuralNetwork::set_weights (const std::vector<double> &w)
{
  std::vector<double>::const_iterator wit;
  wit = w.begin();
  bool end = false;
  
  std::vector<Array<double,2> >::iterator it;
  for (it = weight.begin() ; it != weight.end(); ++it)
    {
      Array<double,2>::iterator iit;
      for (iit = it->begin(); iit != it->end(); ++iit)
	{
	  assert (!end);
	  *iit = *wit;
	  if (wit != w.end()) 
	    {
	      ++wit;
	    }
	  else
	    {
	      end = true;
	    }
	}
    }
}


Array<double,1> NeuralNetwork::inputResponse (const Array<double,1> input)
{
  _fwdPropagation(input);
  return activation.back();
}

void NeuralNetwork::_fwdPropagation(const Array<double,1> input)
{
  activation[0] = input;
  
  bool first_layer = true;
  bool last_layer = false;
  std::vector<Array<double,2> >::const_iterator weight_it;
  std::vector<Array<double,1> >::iterator activation_it;
  activation_it = activation.begin();

  Array<double,1> new_activation;
  unsigned int i = 1;

  for (weight_it = weight.begin(); 
       weight_it != weight.end();
       ++weight_it)
    {
      if (i == weight.size())
	{
	  last_layer = true;
	}
      i++;
	  
      if (!first_layer)
	{
	  (*activation_it)(Range(1,toEnd)) = new_activation; 
	} 
     
      int s = weight_it->extent(secondDim);
      Array<double,1> a(shape(s));
      a = 0;
      new_activation.resize(s);
      a = sum((*weight_it)(tensor::j,tensor::i) * \
	      (*activation_it)(tensor::j), tensor::j);
      if (last_layer)
	{
	  ++activation_it;
	  *activation_it = a; 
	  break;
	}
      else
	{
	  new_activation = tanh(a);
	  first_layer = false;
	  ++activation_it;
	  continue;
	}
    }
}


std::vector<Array<double,1> > NeuralNetwork::_deltas (const Array<double,1> deltaF)
{
  int s1 = deltaF.size();
  int s2 = activation.back().size();
  assert (s1 == s2);  

  unsigned int i = 1;
  bool first_layer = true;
  bool last_layer = false;
  std::vector<Array<double,1> >::const_iterator activation_it;

  std::vector<Array<double,1> > delta;
  // bias units have delta terms
  for (activation_it = activation.begin();
       activation_it != activation.end();
       ++activation_it)
    { 
      if (i==activation.size())
	{
	  last_layer = true;
	}
      i++;
      int s = (*activation_it).size();
      if (!first_layer && !last_layer) 
	{
	  // no delta terms for bias units
	  Array<double,1> d(s-1);
	  delta.push_back(d);
	}
      else if (last_layer)
	{
	  Array<double,1> d(s);
	  delta.push_back(d);
	}
      else {
	first_layer = false;
	continue;
      }
    }
  

  delta.back() = deltaF;

  std::vector<Array<double,2> >::const_iterator weight_it;
  std::vector<Array<double,1> >::iterator delta_it; 

  delta_it = delta.end();
  // place it before last element
  --delta_it;
  i = 1;
  first_layer = false;
  // only for hidden units
  // place before last element
  for (weight_it = --weight.end();
       weight_it != weight.begin();
       --weight_it)
    {
      if (i != weight.size()) {
	int s = weight_it->extent(firstDim);
	Array<double,1> d(shape(s));
	d = sum( (*weight_it)(tensor::i,tensor::j) *	\
		 (*delta_it)(tensor::j), tensor::j);

	--delta_it;
	// we dont need delta terms for bias units
	(*delta_it) = d(Range(1,toEnd));
      }
      i++;
    }

  return delta;
}

Array<double,1> NeuralNetwork::gradientInput(const Array<double,1> input,
					const Array<double,1> deltaF)
{
  _fwdPropagation(input);
  std::vector<Array<double,1> > delta;
  delta = _deltas(deltaF);

  Array<double,1> _dx(layout.front());
  _dx = sum(weight.front()(tensor::i,tensor::j) * \
	    delta.front()(tensor::j), tensor::j);
  
  return _dx;
}



Array<double,1> NeuralNetwork::gradientWeights(const Array<double,1> input,
					const Array<double,1> deltaF)
{
  _fwdPropagation(input);
  std::vector<Array<double,1> > delta;
  delta = _deltas(deltaF);
  
  std::vector<Array<double,1> >::iterator delta_it; 
  std::vector<Array<double,1> >::const_iterator activation_it;
  std::vector<Array<double,2> >::const_iterator weight_it;
  
  activation_it = activation.begin();
  delta_it = delta.begin();
  
  std::vector<double> gradient;
  int bias_unit = 1;
  unsigned int i = 1;
  

  for (weight_it = weight.begin();
       weight_it != weight.end();
       ++weight_it)
    {

      if (i == weight.size())
	{
	  // no biasunit in output layer
	  bias_unit = 0;
	}
      i++;

      Array<double,1> a = (*activation_it);
      ++activation_it;  
      int s = (*activation_it).size() - bias_unit;  
      Array<double,1> da(s);
      
      if (bias_unit == 1) {
	da = dTanh((*activation_it)(Range(1,toEnd)));
      } else {
	// output layer linear activation
	da = dOutAct((*activation_it));
      }
      
      Array<double,1> dda(da.shape());
      dda = (*delta_it)(tensor::i) * da(tensor::i);

      Array<double,2> g((*weight_it).shape());
      g = a(tensor::i) * dda(tensor::j);
 

      Array<double,2>::const_iterator g_it;
      for (g_it = g.begin();
	   g_it != g.end();
	   ++g_it)
	{
	  gradient.push_back(*g_it);
	}
      
      ++delta_it;
      
    }
      
  Array<double,1> gradient_array(gradient.size());
  Array<double,1>::iterator a_it = gradient_array.begin();
  std::vector<double>::const_iterator v_it;
  for (v_it = gradient.begin();
       v_it != gradient.end();
       ++v_it)
    {
      *a_it = *v_it;
      ++a_it;
    }

  return gradient_array;
}

int NeuralNetwork::nWeights()
{
  int n = 0;
  std::vector<Array<double,2> >::iterator it;
  for (it = weight.begin() ; it != weight.end(); ++it)
    {
      Array<double,2>::const_iterator iit;
      for (iit = it->begin(); iit != it->end(); ++iit)
	{
	  n++;
	}
    }
  return n;
}


void NeuralNetwork::clone(const std::shared_ptr<NeuralNetwork> source)
{
     std::vector<Array<double,1> >::const_iterator it;
     for (it = source->activation.begin(); it != source->activation.end(); ++it)
          {
               Array<double,1> a = *it;
               activation.push_back(a);
          }

     std::vector<Array<double,2> >::const_iterator iit;
     for (iit = source->weight.begin(); iit != source->weight.end(); ++iit)
          {
               Array<double,2> w = *iit;
               weight.push_back(w);
          }

     layout = source->layout;
}

