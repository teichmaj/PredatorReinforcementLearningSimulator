
#include "bptt.hpp"
#include "NN/neural_network.hpp"
#include "global.hpp"
#include "config.hpp"
#include "rprop.hpp"
#include "environment.hpp"
#include "sgn.hpp"

#include <blitz/array.h>
#include <cmath> 
#include <csignal>



double nan_guard (double x)
{
  if (std::isnan(x))
    return 1;

  return 0;
}

BZ_DECLARE_FUNCTION(nan_guard)

double big_guard (double x)
{
  if (std::abs(x)> 1e5)
    return 1;
  
  return 0;
}

BZ_DECLARE_FUNCTION(big_guard)


BPTT::BPTT (const Config *CONFIG_,
            ptr_environment env_,
            bool rprop_)
{
  CONFIG = CONFIG_;
  is_rprop = rprop_;
  environment = env_;
  if (is_rprop)
    {
      rprop = new RPROP (CONFIG);
    }
}

BPTT::~BPTT ()
{
  if (is_rprop)
    {
      delete rprop;
    }
}


std::pair<double,double> BPTT::learn
(const  Trajectory &trajectory,
 const std::shared_ptr<NeuralNetwork> Actor,
 const std::shared_ptr<NeuralNetwork> Critic)
{
  
  int num_z = Actor->nWeights();
  Array<double,1> _dJ_dz (num_z);
  _dJ_dz = 0;
  Array<double,1> _dJ_dx_t1 (4);
  _dJ_dx_t1 = 0;

  // final state
  Array<double,1> _state(4);
  Array<double,1> _action(2);
  _state = trajectory.back().first;
  _action = trajectory.back().second;
  _dJ_dx_t1 = environment->dd_state_final_reward(_state);
  

  if (sum(nan_guard(_dJ_dx_t1))>0) {
    std::cout << "BPTT (1) _dJ_dx_t1 " << _dJ_dx_t1;
    std::cout << "BPTT (1) state " << _state;
    std::cout << "BPTT (1) action " << _action;
    raise (SIGINT);
  }

  // zero index offset and second last
  for (int k  = trajectory.size() -2;
       k >0;
       --k)
    {
      _state = trajectory[k].first;
      _action = trajectory[k].second;
      
      // update of _dJdz
      // dJdz = dJdz + gamma^t dAdz ( dUdu + gamma dfdu dJdx_t1 )
      //                                     ------ (1) ------
      //                            ----------- (2) _dU ----------
      //              ------------------ (3) ---------------------
      // (4) -----------------------------------------------------

      // (1)
      Array<double, 2> _df_du(4,2);
      _df_du = environment->dd_action_dot_state(_action);
	
      if (sum(nan_guard(_df_du))>0) {
	std::cout << "BPTT (2) _df_du " << _df_du;
	std::cout << "BPTT (2) state " << _state;
	std::cout << "BPTT (2) action " << _action;
	raise (SIGINT);
      }

      Array<double,1> _dfJ_dux (2);
      _dfJ_dux = sum(_df_du(tensor::j,tensor::i) *	\
                     _dJ_dx_t1(tensor::j), tensor::j);
      _dfJ_dux *= CONFIG->gamma;
      
      if (sum(nan_guard(_dfJ_dux))>0) {
	std::cout << "BPTT (3) _dfJ_dux " << _dfJ_dux;
	std::cout << "BPTT (3) _df_da " << _df_du;
	std::cout << "BPTT (3) _dJ_dx_t1 " << _dJ_dx_t1;
	std::cout << "BPTT (3) state " << _state;
	std::cout << "BPTT (3) action " << _action;
	raise(SIGINT);
      }
 
      
      // (2)
      Array<double,1> _dU_du (2);
      _dU_du = environment->dd_action_energy(_action);
      
      if (sum(nan_guard(_dU_du))>0) {
	std::cout << "BPTT (4) _dU_du " << _dU_du;
	std::cout << "BPTT (4) state " << _state;
	std::cout << "BPTT (4) action " << _action;
	raise(SIGINT);
      }

      Array<double,1> _dU (2);
      _dU =  _dU_du + _dfJ_dux;
      
      if (sum(nan_guard(_dU))>0) {
	std::cout << "BPTT (5) _dU " << _dU;
	std::cout << "BPTT (5) _dU_du " << _dU_du;
	std::cout << "BPTT (5) _dfJ_dux " << _dfJ_dux;
	std::cout << "BPTT (5) state " << _state;
	std::cout << "BPTT (5) action " << _action;
	raise(SIGINT);
      }
      
      // (3)
      Array<double,1> _dA_dz (num_z);
      _dA_dz = Actor->gradientWeights(_state, _dU);
      _dA_dz *= pow(CONFIG->gamma, double(k));
      
      if (sum(nan_guard(_dA_dz))>0) {
	std::cout << "BPTT (6) dA_dz " <<  _dA_dz;
	std::cout << "BPTT (6) _dU " << _dU;
	std::cout << "BPTT (6) _dU_du " << _dU_du;
	std::cout << "BPTT (6) _dfJ_dux " << _dfJ_dux;
	std::cout << "BPTT (6) _df_du " << _df_du;
	std::cout << "BPTT (6) _dJ_dx_t1 " << _dJ_dx_t1;
	std::cout << "BPTT (6) state " << _state;
	std::cout << "BPTT (6) action " << _action;
	raise (SIGINT);
      }


      // (4)
      _dJ_dz += _dA_dz;

      if (sum(nan_guard(_dJ_dz))>0) {
	std::cout << "BPTT (7) _dJ_dz " <<  _dJ_dz;
	std::cout << "BPTT (7) iteration " << k;
	std::cout << "BPTT (7) state " << _state;
	std::cout << "BPTT (7) action " << _action;
	raise(SIGINT);
      }

      //
      // ****************************************************
      //


      // update of dJdx_t
      // dJdx_t = dUdx + gamma dfdx dJdx_t1 + dAdx (dUdu + gamma dfdu dJdx_t1 )
      //                                           --------- _dU --------------
      //                                      ---------------- (1) ------------
      //         -(3)-   ------- (2) ------
      // (4) ------------------------------------------------------------------


      // (1)
      Array<double,1> _dA_dx(4);
      _dA_dx = Actor->gradientInput(_state, _dU);

      if (sum(nan_guard(_dA_dx))>0) {
	std::cout << "\n\n";
	std::cout << "BPTT (8) _dA_dx " <<  _dA_dx;
	std::cout << "BPTT (8) _dU " <<  _dU;
	std::cout << "BPTT (8) _df_du " <<  _df_du;
	std::cout << "BPTT (8) _dJ_dx_t1 " <<  _dJ_dx_t1;
	std::cout << "BPTT (8) state " << _state;
	std::cout << "BPTT (8) action " << _action;
	raise(SIGINT);
      }

      // (2)
     Array<double,2> _df_dx (4,4);
      _df_dx = environment->dd_state_dot_state(_state);
      
      if (sum(nan_guard(_df_dx))>0) {
	std::cout << "BPTT (9) _df_dx " <<  _df_dx;
	std::cout << "BPTT (9) state " << _state;
	std::cout << "BPTT (9) action " << _action;
	raise(SIGINT);
      }

      Array<double,1> _dfJ_dx (4);
      _dfJ_dx = sum(_df_dx(tensor::i,tensor::j) *	\
		    _dJ_dx_t1(tensor::j), tensor::j);
      _dfJ_dx *= CONFIG->gamma;
    
      if (sum(nan_guard(_dfJ_dx))>0) {
	std::cout << "BPTT (10) _dfJ_dx " <<  _dfJ_dx;
	std::cout << "BPTT (10) state " << _state;
	std::cout << "BPTT (10) action " << _action;
	raise(SIGINT);
      }

      
      // (3)
      Array<double,1> _dU_dx (4);
      _dU_dx = environment->dd_state_energy(_state);
      
      if (sum(nan_guard(_dU_dx))>0) {
	std::cout << "BPTT (11) _dU_dx " <<  _dU_dx;
	std::cout << "BPTT (11) state " << _state;
	std::cout << "BPTT (11) action " << _action;
	raise(SIGINT);
      }

      // (4)
      Array<double,1> _dJ_dx_t (4);
      _dJ_dx_t = _dU_dx + _dfJ_dx + _dA_dx;
      
      if (sum(nan_guard(_dJ_dx_t))>0) {
	std::cout << "BPTT (12) _dJ_dx_t " <<  _dJ_dx_t;
	std::cout << "BPTT (12) _dfJ_dx " <<  _dfJ_dx;
	std::cout << "BPTT (12) _dA_dx " <<  _dA_dx;
	std::cout << "BPTT (12) _dU_dx " <<  _dU_dx;
	std::cout << "BPTT (12) state " << _state;
	std::cout << "BPTT (12) action " << _action;
	raise(SIGINT);
      }
      
      _dJ_dx_t1 = _dJ_dx_t;
    }
   

  std::vector<double> _weights;
  _weights = Actor->get_weights_flattened();
  std::vector<double>::iterator z_it;
  Array<double,1>::const_iterator u_it;

  double delta;
  double alpha;
  if (is_rprop)
    {
      Array<double,1> _rprop_dz(num_z);
      _rprop_dz = rprop->opt(_dJ_dz);
      delta = sum(abs(_rprop_dz));
      u_it = _rprop_dz.begin();
      alpha = 1;
    }
  else
    {
      delta = sum(abs(_dJ_dz));
      u_it = _dJ_dz.begin();  
      alpha = CONFIG->alpha;
    }
  
  for (z_it = _weights.begin();
       z_it != _weights.end();
       ++z_it)
    {
      double _z_i = *z_it;
      double _u_i = *u_it;
      *z_it =  _z_i + (alpha * _u_i);
      ++u_it;
    }

  
  Actor->set_weights(_weights);
  return std::make_pair(delta,0.0);
}



bool BPTT::test (const std::shared_ptr<NeuralNetwork> NN,
                 const std::shared_ptr<NeuralNetwork> Critic)
{
  bool all_pass = true;
  int num_z = NN->nWeights();

  
  Trajectory trajectory;
  Array<double,1> state(4);
  state(TIME) = 10;
  state(AGE) = 0.01;
  state(D1) = 2;
  state(D2) = 2;
  Array<double,1> action(2);
  action(dE1) = 1;
  action(dE2) = -1;
  for (int i=0; i<4; i++) {
    trajectory.push_back(std::make_pair(state.copy(), action.copy()));
  }
  state(TIME) = CONFIG->length_of_episode + 1.0;
  trajectory.push_back(std::make_pair(state.copy(), action.copy()));

   Array<double,1> _dJ_dx_t1 (4);
  _dJ_dx_t1 = environment->dd_state_final_reward(state);

  std::cout << "BPTT Test (1): dFState_dstate\n--------------------------------------------\n";
  std::cout << "state: " << state;
  std::cout << "dist to home: " << environment->dist_to_home(state) << std::endl;
  std::cout << "dJ_dx_t1: " << _dJ_dx_t1;
 

  std::cout << "\n\n";
  std::cout << "###################################\nTRAJECTORY ROLL-BACK\n###################################\n\n";


  Array<double,1> _dJ_dz(num_z);
  _dJ_dz = 0;

  for (int kk= trajectory.size() -2;
       kk > 0;
       --kk)
    {
      state = trajectory[kk].first;
      action = trajectory[kk].second;

      std::cout << "########\n[ " << kk << "]\n########\n";
      std::cout << "    state: " << state;
      std::cout << "    action: " << action;

  
   Array<double, 2> _df_du(4,2);
   _df_du = environment->dd_action_dot_state(action);
   Array<double,1> _dfJ_dux (2);
   _dfJ_dux = sum(_df_du(tensor::j,tensor::i) *         \
                  _dJ_dx_t1(tensor::j), tensor::j);
   _dfJ_dux *= CONFIG->gamma;
   
   std::cout << "BPTT Test (2): gamma * dfdu * dJdxt1\n--------------------------------------------\n";
   std::cout << "state: " << state;
   std::cout << "action: " << action;
   std::cout << "gamma: " << CONFIG->gamma <<std::endl;
   std::cout << "dfdu: " << _df_du;
   std::cout << "dJdxt1: " << _dJ_dx_t1;
   std::cout << "dfJ_dux: " << _dfJ_dux;
   std::cout << "\n\n";




   Array<double,1> _dU_du (2);
   _dU_du = environment->dd_action_energy(action);
   Array<double,1> _dU (2);
   _dU =  _dU_du + _dfJ_dux;
    std::cout << "BPTT Test (3): dU = dUdu + (2)\n--------------------------------------------\n";
   std::cout << "action: " << action;
   std::cout << "dUdu: " << _dU_du;
   std::cout << "result: " << _dU;
   std::cout << "\n\n";


   int k = 5;
   std::vector<double> weights;
   weights = NN->get_weights_flattened();
   Array<double,1> _dA_dz (num_z);
   _dA_dz = NN->gradientWeights(state, _dU);
   _dA_dz *= pow(CONFIG->gamma, double(k));

   std::cout << "BPTT Test (4): gamma ^ k * dAdz * dU (3)\n--------------------------------------------\n";
   std::cout << "state: " << state;
   std::cout << "dU: " << _dU;
   std::cout << "gamma ^" << k <<": " <<  pow(CONFIG->gamma, double(k)) << std::endl;
   std::cout << "result:\n\tweights\t\td_weights\n------------------------------------\n";
   std::cout << std::setprecision(6) << std::fixed;
   for (int i=0; i<num_z; i++) {
     std::cout << "["<<i<<"]\t" << weights[i] << "\t" << double(_dA_dz(i)) << std::endl;
   }
   std::cout << "NN response: " << NN->inputResponse(state);  
   std::cout << "\n\n";



   Array<double,1> _dA_dx(4);
   _dA_dx = NN->gradientInput(state, _dU);
    std::cout << "BPTT Test (5): dAdx * dU (3)\n--------------------------------------------\n";
   std::cout << "dU: " << _dU;
   std::cout << "result:\n\tstate\t\td_state\n------------------------------------\n";
   std::cout << std::setprecision(6) << std::fixed;
   for (int i=0; i<4; i++) {
     std::cout <<  "["<<i<<"]\t" << state(i) << "\t" << double(_dA_dx(i)) << std::endl;
   }
  std::cout << "NN response: " << NN->inputResponse(state);  
  std::cout << "\n\n";

 

   Array<double,2> _df_dx (4,4);
   _df_dx = environment->dd_state_dot_state(state);
    Array<double,1> _dfJ_dx (4);
    _dfJ_dx = sum(_df_dx(tensor::i,tensor::j) *	\
                  _dJ_dx_t1(tensor::j), tensor::j);
    _dfJ_dx *= CONFIG->gamma;
   std::cout << "BPTT Test (6): gamma * dfdx * dJdx_t1\n--------------------------------------------\n";
   std::cout << "gamma: " << CONFIG->gamma << std::endl;
   std::cout << "state: " << state;
   std::cout << "dfdx: " << _df_dx;
   std::cout << "dJdx_t1: " << _dJ_dx_t1;
   std::cout << "dfJ_dx: " << _dfJ_dx;
   std::cout << "\n\n";
     

   Array<double,1> _dU_dx (4);
   _dU_dx = environment->dd_state_energy(state);
    std::cout << "BPTT Test (7): dUdx\n--------------------------------------------\n";
    std::cout << "state: " << state;
    std::cout << "dU_dx: " << _dU_dx;
    std::cout << "\n\n";

    
    Array<double,1> _dJdx_t(4);
    _dJdx_t = _dU_dx + _dfJ_dx + _dA_dx;
    _dJ_dx_t1 = _dJdx_t;
    _dJ_dz += _dA_dz;
    std::cout << "BPTT Test (8): Total Results:\n######################################################\n######################################################\n";
    std::cout << "dJdx_t = dUdx + dfJdx + dAdx\n";
    std::cout << "dUdx: " << _dU_dx;
    std::cout << "dfJdx: " << _dfJ_dx;
    std::cout << "dAdx: " << _dA_dx;
    std::cout << "dJdx_t: " << _dJdx_t;
    std::cout << "weight update: \n";
    std::cout << std::setprecision(6) << std::fixed;
     std::cout << "\tweights\t\tdelta\t\ttotal_delta\n---------------------------------------------------------\n";
    for (int i=0; i<num_z; i++) {
      std::cout << "["<<i<<"]\t" << weights[i] << "\t" << double(_dA_dz(i)) << "\t" << double(_dJ_dz(i)) << std::endl;
    }
    std::cout << "\n######################################################\n######################################################\n\n";

    }

  return all_pass;
}


