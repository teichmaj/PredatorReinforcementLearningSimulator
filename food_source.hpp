#ifndef _FOOD_SOURCE_H_
#define _FOOD_SOURCE_H_

#include "config.hpp"
#include "global.hpp"

#include <memory>


class IFoodSource {
public:
	virtual ~IFoodSource(){;};
	virtual double get_reward (const Array<double,1>)=0;
	virtual double get_time (const Array<double,1>)=0;
	virtual Array<double,1> dd_R_action (const Array<double,1>)=0;
	virtual Array<double,2> dd_T_action (const Array<double,1>)=0;
	virtual Array<double,1> dd_R_state (const Array<double,1>)=0;
	virtual Array<double,2> dd_T_state (const Array<double,1>)=0;
	virtual bool test (const Array<double,1>,
					   const Array<double,1>)=0;
	virtual Array<double,1> location ()=0;
private:
  IFoodSource& operator=(const IFoodSource&);
};


template<typename F>
class FoodSource : public IFoodSource {
public:
	FoodSource (const Config *CONFIG_,
				const F &enc_f_,
				const double tox_,
				const double rew_,
				const double handling_,
				const double p_): 
		Encounter(enc_f_), CONFIG(CONFIG_), tox(tox_),
		rew(rew_), t_handling(handling_), p(p_) {};

	virtual ~FoodSource(){;};

	virtual double get_reward (const Array<double,1> state)
	{
		double enc_freq = Encounter.enc_freq(state);
		double s = taste_sampling();
		double payout = reward_payout();
		double result;
		result = enc_freq * s * payout * p; 
		
		return result;
	};

	
  virtual double get_time (const Array<double,1> state)
	{
		double enc_freq = Encounter.enc_freq(state);
		double s = taste_sampling();
		double t_sam = taste_sampling_time();
		double t_tox = toxin_recovery();
		double result;
		result = enc_freq * p * ( s*(t_handling+t_tox) + t_sam );
		return result;
	};

  
	virtual Array<double,1> dd_R_action (const Array<double,1> action)
	{
		Array<double,1> result(2);
		result = 0;
		return result;
	};

	virtual Array<double,2> dd_T_action (const Array<double,1> action)
	{
		Array<double,2> result(4,2);
		result = 0;
		return result;
	};

  virtual Array<double,1> dd_R_state (const Array<double,1> state)
	{
		double s = taste_sampling();
		Array<double,1> result(4);
		double payout = reward_payout();
		result = 0;
		result = Encounter.dd_state(state) * s * payout * p;
		return result;
	};

	virtual Array<double,2> dd_T_state (const Array<double,1> state)
	{
		Array<double,2> result(4,4);
		result = 0;
		
		double s = taste_sampling();
		double t_sam = taste_sampling_time ();
		double t_tox = toxin_recovery();
		
		result(TIME,Range::all()) = Encounter.dd_state(state) * \
			(t_sam + s*(t_handling + t_tox) ) * p;
		
		return result;
	};

  virtual bool test (const Array<double,1> state,
					 const Array<double,1> action)
	{
		double h = 0.0001;
		bool all_pass = true;
		
		// state derivative REWARD
		//
		Array<double,1> _hs(4);
		Array<double,1> res_st(4);
		
		for (int i=0; i<4; i++)
			{
				_hs = 0;
				_hs(i) = h;
				Array<double,1> s1(4);
				s1 = state + _hs;
				Array<double,1> s2(4);
				s2 = state - _hs;
				res_st(i) = (get_reward(s1)-get_reward(s2)) / (2*h);
			}
		
		Array<double,1> dState (4);
		dState = dd_R_state(state);
		if ( sum(abs(res_st-dState)) > 0.001) {
			all_pass = false;
			std::cout << "FoodSource::test() state derivative of REWARD failed!\n";
			 std::cout << "derivative: " << dState << std::endl;
			 std::cout << "stencil: " << res_st << std::endl;
		}

		// state derivative TIME
		//
		Array<double,2> res_st2(4,4);
		res_st2 = 0;
		for (int i=0; i<4; i++)
			{
				_hs = 0;
				_hs(i) = h;
				Array<double,1> s1(4);
				s1 = state + _hs;
				Array<double,1> s2(4);
				s2 = state - _hs;
				res_st2(TIME,i) = (get_time(s1)-get_time(s2)) / (2*h);
			}
		
		Array<double,2> dState2 (4,4);
		dState2 =  dd_T_state(state);
		if ( sum(abs(res_st2-dState2)) > 0.001) {
		  all_pass = false;
		  std::cout << "FoodSource::test() state derivative of TIME failed!\n";
		  std::cout << "derivative: " << dState2 << std::endl;
		  std::cout << "stencil: " << res_st2 << std::endl;
		}
		
		return all_pass;
	};
  
  virtual Array<double,1> location ()
  {
    return Encounter.location();
  }
  
	
private:
  
  double taste_sampling ()
  {
    return 1.0 / (1.0 + CONFIG->s_0 * tox);
  };
  
  double taste_sampling_time ()
  {
    double result = 0;
    if (tox > 0)
      {
	result = CONFIG->time_sampling;
      }
    return result;
  };
  
  double toxin_recovery ()
  {
    return CONFIG->time_toxin * tox * tox;
  };
  
  double reward_payout ()
  {
    double payout = rew - (tox * tox);
    return payout;
  }
  
  
  F Encounter;
  const Config * CONFIG;
  const double tox;
  const double rew;
  const double t_handling;
  double p;	
};


typedef std::shared_ptr<IFoodSource> ptr_food_source;
typedef std::vector<ptr_food_source> FoodSources;





#endif
