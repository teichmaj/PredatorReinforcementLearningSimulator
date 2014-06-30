#include "vgl.hpp"
#include "NN/neural_network.hpp"
#include "global.hpp"
#include "config.hpp"
#include "rprop.hpp"
#include "environment.hpp"
#include "sgn.hpp"

#include <cmath> 
#include <csignal>
#include <blitz/array.h>

using namespace blitz;


VGL::VGL (const Config *CONFIG_,
			  ptr_environment env_,
			  bool rprop_)
{
  CONFIG = CONFIG_;
  is_rprop = rprop_;
  environment = env_;
  if (is_rprop)
    {
      rprop_z = new RPROP (CONFIG);
      rprop_w = new RPROP (CONFIG);
    }
}


VGL::~VGL ()
{
  if (is_rprop)
    {
      delete rprop_z;
delete rprop_w;
    }
}


std::pair<double,double> VGL::learn
(const  Trajectory &trajectory,
 const std::shared_ptr<NeuralNetwork> Actor,
 const std::shared_ptr<NeuralNetwork> Critic)
{
  double lambda = CONFIG->lambda_vgl;
  double gamma = CONFIG->gamma;
  int num_w = Critic->nWeights();
  int num_z = Actor->nWeights();

  Array<double,1> _delta_w(num_w);
  _delta_w = 0;
  Array<double,1> _delta_z(num_z);
  _delta_z = 0;

  Array<double,1> _state(4);
  Array<double,1> _action(2);
  _state = trajectory.back().first;
  _action = trajectory.back().second;

  Array<double, 1> _Gdash_t1(4);
  _Gdash_t1 = environment->dd_state_final_reward(_state);
  Array<double,1> _Gtilde_t1(4);
  _Gtilde_t1 = Critic->inputResponse(_state);

  Array<double,1> _G_error(4);
  _G_error = _Gdash_t1 - _Gtilde_t1;
  _delta_w = Critic->gradientWeights(_state, _G_error);


  // ######################
  // Trajectory backpass
  // ######################
  for (int k  = trajectory.size() -2;
       k >0;
       --k)
    {
      _state = trajectory[k].first;
      _action = trajectory[k].second;
  
      Array<double,1> _p(4);
      _p = (lambda * _Gdash_t1) +               \
        ((1.0-lambda) * _Gtilde_t1);

      // G' = dUdx + gamma dfdx p + dAdx (dUdu + gamma dfdu p)
      //                                         ---- (1) ----
      //                                 -------- (2) --------
      //                            ------ (3) ---------------
      //             ---- (4) ----
      //    -------------------- (5) -------------------------
      
      // (1)
      Array<double,2> _df_du(4,2);
      _df_du = environment->dd_action_dot_state(_action);
      Array<double,1> _g_df_du_p(2);
      _g_df_du_p = sum(_df_du(tensor::j,tensor::i) *    \
                       _p(tensor::j), tensor::j);
      _g_df_du_p *= gamma;
      
      
      // (2)
      Array<double,1> _dU_du(2);
      _dU_du = environment->dd_action_energy(_action);
      Array<double,1> _dU_df_du_p(2);
      _dU_df_du_p = _g_df_du_p + _dU_du;
      
      
      // (3)
      Array<double,1> _dA_dx(4);
      _dA_dx = Actor->gradientInput(_state, _dU_df_du_p);
      

      // (4)
      Array<double,2> _df_dx(4,4);
      _df_dx = environment->dd_state_dot_state(_state);
      Array<double,1> _g_df_dx_p(4);
      _g_df_dx_p = sum(_df_dx(tensor::i, tensor::j) *   \
                       _p(tensor::j), tensor::j);
      _g_df_dx_p *= gamma;
      
      
      // (5)
      Array<double,1> _dU_dx(4);
      _dU_dx = environment->dd_state_energy(_state);
      Array<double,1> _Gdash_t(4);
      _Gdash_t = _dU_dx + _g_df_dx_p + _dA_dx;
      


      
      // dw = dw + dG~dw Omega (G'_t - G~_t)
      //                       ---- (1) ----
      //           ------ (2) --------------
      // (3) -------------------------------

      // (1)
      Array<double,1> _Gtilde_t(4);
      _Gtilde_t = Critic->inputResponse(_state);
      _G_error = _Gdash_t - _Gtilde_t;
      
      // (2)
      Array<double,1> _dG_dw_Gerror (num_w);
      _dG_dw_Gerror = Critic->gradientWeights(_state, _G_error);
      
      // (3)
      _delta_w += _dG_dw_Gerror;

      



      // dz = dz - dAdz (dUdu + gamma (dfdu G~_t+1))
      //                        -------- (1) -------
      //                 -------- (2) --------------
      //           --------------- (3) -------------
      // (4) ---------------------------------------
      
      
      // (1)
      Array<double,1> _g_df_du_Gtilde(2);
      _g_df_du_Gtilde = sum(_df_du(tensor::j, tensor::i) *  \
                            _Gtilde_t1(tensor::j), tensor::j);
      _g_df_du_Gtilde *= gamma;
      
      // (2)
      Array<double,1> _dU_df_du_G (2);
      _dU_df_du_G = _dU_du + _g_df_du_Gtilde;
      
      // (3)
      Array<double,1> _dA_dz(num_z);
      _dA_dz = Actor->gradientWeights(_state, _dU_df_du_G);
      
      // (4)
      _delta_z += _dA_dz;
      

      
      // discounted back propagation
      // ###########################

      _Gtilde_t1 = Critic->inputResponse(_state);
      _Gdash_t1 = _Gdash_t;

    }


  std::vector<double> _weights_w;
  _weights_w = Critic->get_weights_flattened();
  std::vector<double>::iterator w_it;
  Array<double,1>::const_iterator dw_it;
  double Delta_w;
  double alpha;
  if (is_rprop)
    {
      Array<double,1> _rprop_dw(num_w);
      _rprop_dw = rprop_w->opt(_delta_w);
      Delta_w = sum(abs(_rprop_dw));
      dw_it = _rprop_dw.begin();
      alpha = 1;
    }
  else
    {
      Delta_w = sum(abs(_delta_w));
      dw_it = _delta_w.begin();  
      alpha = CONFIG->alpha;
    }
  
  for (w_it = _weights_w.begin();
       w_it != _weights_w.end();
       ++w_it)
    {
      double _w_i = *w_it;
      double _dw_i = *dw_it;
      *w_it =  _w_i + (alpha * _dw_i);
      ++dw_it;
    }




  

  std::vector<double> _weights_z;
  _weights_z = Actor->get_weights_flattened();
  std::vector<double>::iterator z_it;
  Array<double,1>::const_iterator dz_it;
  double Delta_z;
  if (is_rprop)
    {
      Array<double,1> _rprop_dz(num_z);
      _rprop_dz = rprop_z->opt(_delta_z);
      Delta_z = sum(abs(_rprop_dz));
      dz_it = _rprop_dz.begin();
      alpha = 1;
    }
  else
    {
      Delta_z = sum(abs(_delta_z));
      dz_it = _delta_z.begin();  
      alpha = CONFIG->alpha;
    }
  
  for (z_it = _weights_z.begin();
       z_it != _weights_z.end();
       ++z_it)
    {
      double _z_i = *z_it;
      double _dz_i = *dz_it;
      *z_it =  _z_i + (alpha * _dz_i);
      ++dz_it;
    }


  Actor->set_weights(_weights_z);
  Critic->set_weights(_weights_w);
  
   
// Actor, Critic
  return std::make_pair(Delta_z, Delta_w);
}


bool VGL::test (const std::shared_ptr<NeuralNetwork> Actor,
                 const std::shared_ptr<NeuralNetwork> Critic)
{
  bool all_pass = true;

  return all_pass;
}
