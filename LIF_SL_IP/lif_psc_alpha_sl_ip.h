/*
 *  lif_psc_alpha_sl_ip.h
 *
 *  This file is part of NEST.
 *
 *  Copyright (C) 2004 The NEST Initiative
 *
 *  NEST is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  NEST is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with NEST.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#ifndef LIF_PSC_ALPHA_SL_IP_H
#define LIF_PSC_ALPHA_SL_IP_H

// Includes from nestkernel:
#include "archiving_node.h"
#include "connection.h"
#include "event.h"
#include "nest_types.h"
#include "ring_buffer.h"
#include "universal_data_logger.h"

// Includes from sli:
#include "dictdatum.h"

namespace mynest
{

/* BeginDocumentation
Name: lif_psc_alpha_sl_ip - Leaky integrate-and-fire neuron model with alpha PSC
                      synapse, spike latency and intrinsic plasticity.

Description:
lif_psc_alpha_sl_ip implements a leaky integrate-and-fire neuron with
with alpha-function shaped synaptic currents. Also, the neuron exhibit a spike latency
avobe the threshold. Therefore, the neuron model modify the sinaptic input value thanks
to a intrinsic plasticity which depends on spike times. This neuron model facilitate the
detection of spike patterns.

Parameters:
C_m      double - Membrane capacitance, in pF
I_e      double - Intrinsic DC current, in nA
tau_syn  double - Synaptic time constant, in ms
t_ref    double - Duration of refractory period in ms.
V_th     double - Spike threshold in mV.
V_reset  double - Reset potential of the membrane in mV.
Decay    double - Decay of the voltage due to a leaky characteristic.
nueromodulator array - array with GID of neurons which modulate the Intrinsic plasticity.
lambda   double . Value of change per intrinsic plasticity effect.
tau      double . value of window of the intrinsic plasticity effect.
std_mod  bool   . Swhich ON (true) or OFF (false) the intrinsic plasticity effect.


Sends: SpikeEvent

Receives: SpikeEvent, CurrentEvent, DataLoggingRequest

Author:
Alejandro Santos Mayo, based on iaf_psc_alpha

*/

/**
 * Non-leaky integrate-and-fire neuron with alpha-shaped PSCs.
 */
class lif_psc_alpha_sl_ip : public nest::Archiving_Node
{
public:
  /**
   * The constructor is only used to create the model prototype in the model
   * manager.
   */
  lif_psc_alpha_sl_ip();

  /**
   * The copy constructor is used to create model copies and instances of the
   * model.
   * @node The copy constructor needs to initialize the parameters and the
   * state.
   *       Initialization of buffers and interal variables is deferred to
   *       @c init_buffers_() and @c calibrate().
   */
  lif_psc_alpha_sl_ip( const lif_psc_alpha_sl_ip& );

  /**
   * Import sets of overloaded virtual functions.
   * This is necessary to ensure proper overload and overriding resolution.
   * @see http://www.gotw.ca/gotw/005.htm.
   */
  using nest::Node::handle;
  using nest::Node::handles_test_event;

  /**
   * Used to validate that we can send SpikeEvent to desired target:port.
   */
  nest::port send_test_event( nest::Node&, nest::port, nest::synindex, bool );

  /**
   * @defgroup mynest_handle Functions handling incoming events.
   * We tell nest that we can handle incoming events of various types by
   * defining @c handle() and @c connect_sender() for the given event.
   * @{
   */
  void handle( nest::SpikeEvent& );         //! accept spikes
  void handle( nest::CurrentEvent& );       //! accept input current
  void handle( nest::DataLoggingRequest& ); //! allow recording with multimeter

  nest::port handles_test_event( nest::SpikeEvent&, nest::port );
  nest::port handles_test_event( nest::CurrentEvent&, nest::port );
  nest::port handles_test_event( nest::DataLoggingRequest&, nest::port );
  /** @} */

  void get_status( DictionaryDatum& ) const;
  void set_status( const DictionaryDatum& );

private:
  //! Reset parameters and state of neuron.

  //! Reset state of neuron.
  void init_state_( const Node& proto );

  //! Reset internal buffers of neuron.
  void init_buffers_();

  //! Initialize auxiliary quantities, leave parameters and state untouched.
  void calibrate();

  //! Take neuron through given time interval
  void update( nest::Time const&, const long, const long );

  // The next two classes need to be friends to access the State_ class/member
  friend class nest::RecordablesMap< lif_psc_alpha_sl_ip >;
  friend class nest::UniversalDataLogger< lif_psc_alpha_sl_ip >;

  /**
   * Free parameters of the neuron.
   *
   * These are the parameters that can be set by the user through @c SetStatus.
   * They are initialized from the model prototype when the node is created.
   * Parameters do not change during calls to @c update() and are not reset by
   * @c ResetNetwork.
   *
   * @note Parameters_ need neither copy constructor nor @c operator=(), since
   *       all its members are copied properly by the default copy constructor
   *       and assignment operator. Important:
   *       - If Parameters_ contained @c Time members, you need to define the
   *         assignment operator to recalibrate all members of type @c Time .
   *         You may also want to define the assignment operator.
   *       - If Parameters_ contained members that cannot copy themselves, such
   *         as C-style arrays, you need to define the copy constructor and
   *         assignment operator to copy those members.
   */
  struct Parameters_
  {
    double C_m;     //!< Membrane capacitance, in pF.
    double I_e;     //!< Intrinsic DC current, in nA.
    double tau_syn; //!< Synaptic time constant, in ms.
    double V_th;    //!< Spike threshold, in mV.
    double V_reset; //!< Reset potential of the membrane, in mV.
    double t_ref;   //!< Duration of refractory period, in ms.
    long stepms;    //!< PASO DE CADA LAP in ms. para calcular next tf
	double V_umbral; //!< Umbral sobre el cual empieza el spike latency
	double Decay;   //////////////////////////////////////////////////////////////////!!!! DECAIMIENTO por cada ms en el leakage phase
	double dt;   /////////////////////////////////////////////////////////////////////!!!! DRESOLUCION DE TIEMPO.
	std::vector< long > neuromodulator; //////////////////////////////////////////////////////////!!!!  GID de la NEURONA NEUROMODULADORA   int
	std::vector< double > prev_w;
	double lambda;
	double tau;
	bool std_mod;
	//bool std_mod;
    //! Initialize parameters to their default values.
    Parameters_();

    //! Store parameter values in dictionary.
    void get( DictionaryDatum& ) const;

    //! Set parameter values from dictionary.
    void set( const DictionaryDatum& );
  };

  /**
   * Dynamic state of the neuron.
   *
   * These are the state variables that are advanced in time by calls to
   * @c update(). In many models, some or all of them can be set by the user
   * through @c SetStatus. The state variables are initialized from the model
   * prototype when the node is created. State variables are reset by @c
   * ResetNetwork.
   *
   * @note State_ need neither copy constructor nor @c operator=(), since
   *       all its members are copied properly by the default copy constructor
   *       and assignment operator. Important:
   *       - If State_ contained @c Time members, you need to define the
   *         assignment operator to recalibrate all members of type @c Time .
   *         You may also want to define the assignment operator.
   *       - If State_ contained members that cannot copy themselves, such
   *         as C-style arrays, you need to define the copy constructor and
   *         assignment operator to copy those members.
   */
  struct State_
  {
    double V_m;      //!< Membrane potential, in mV.
    double dI_syn;   //!< Derivative of synaptic current, in nA/ms.
    double I_syn;    //!< Synaptic current, in nA.
    double I_ext;    //!< External current, in nA.
    long refr_count; //!< Number of steps neuron is still refractory for
	double Vpositivo; //!< ***************** valor positivo +70 para calucluar luego
	double tf; //!< ***************** valor time to fire calculado en presente
	double tfsiguiente; //!< ***************** valor del time to fire del siguiente step 
	double Vactualizado; //!< ***************** valor positivo +70 para calucluar luego
	double enhancement; /////////////////////////////////////////////////////////////////////////////////////////////////////////////// enhancement
	std::vector< double > previous_w;  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////  previous w
	double suma;
	double t_lastspike_;
	double Kminus2_;
	
    /**
     * Construct new default State_ instance based on values in Parameters_.
     * This c'tor is called by the no-argument c'tor of the neuron model. It
     * takes a reference to the parameters instance of the model, so that the
     * state can be initialized in accordance with parameters, e.g.,
     * initializing the membrane potential with the resting potential.
     */
    State_( const Parameters_& );

    /** Store state values in dictionary. */
    void get( DictionaryDatum& ) const;

    /**
     * Set membrane potential from dictionary.
     * @note Receives Parameters_ so it can test that the new membrane potential
     *       is below threshold.
     */
    void set( const DictionaryDatum&, const Parameters_& );
  };

  /**
   * Buffers of the neuron.
   * Ususally buffers for incoming spikes and data logged for analog recorders.
   * Buffers must be initialized by @c init_buffers_(), which is called before
   * @c calibrate() on the first call to @c Simulate after the start of NEST,
   * ResetKernel or ResetNetwork.
   * @node Buffers_ needs neither constructor, copy constructor or assignment
   *       operator, since it is initialized by @c init_nodes_(). If Buffers_
   *       has members that cannot destroy themselves, Buffers_ will need a
   *       destructor.
   */
  struct Buffers_
  {
    Buffers_( lif_psc_alpha_sl_ip& );
    Buffers_( const Buffers_&, lif_psc_alpha_sl_ip& );

    nest::RingBuffer spikes;   //!< Buffer incoming spikes through delay, as sum
    nest::RingBuffer currents; //!< Buffer incoming currents through delay,
                               //!< as sum

    //! Logger for all analog data
    nest::UniversalDataLogger< lif_psc_alpha_sl_ip > logger_;
  };

  /**
   * Internal variables of the neuron.
   * These variables must be initialized by @c calibrate, which is called before
   * the first call to @c update() upon each call to @c Simulate.
   * @node Variables_ needs neither constructor, copy constructor or assignment
   *       operator, since it is initialized by @c calibrate(). If Variables_
   *       has members that cannot destroy themselves, Variables_ will need a
   *       destructor.
   */
  struct Variables_
  {
    double P11;
    double P21;
    double P22;
    double P31;
    double P32;
    double P30;
    double P33;

    double pscInitialValue;
    long t_ref_steps; //!< Duration of refractory period, in steps.
  };

  /**
   * @defgroup Access functions for UniversalDataLogger.
   * @{
   */
  //! Read out the real membrane potential
  double
  get_V_m_() const
  {
    return S_.V_m;
  }
  double
  get_soma_exc_() const
  {
    return S_.enhancement;
  }
  /** @} */

  /**
   * @defgroup pif_members Member variables of neuron model.
   * Each model neuron should have precisely the following four data members,
   * which are one instance each of the parameters, state, buffers and variables
   * structures. Experience indicates that the state and variables member should
   * be next to each other to achieve good efficiency (caching).
   * @note Devices require one additional data member, an instance of the @c
   *       Device child class they belong to.
   * @{
   */
  Parameters_ P_; //!< Free parameters.
  State_ S_;      //!< Dynamic state.
  Variables_ V_;  //!< Internal Variables
  Buffers_ B_;    //!< Buffers.

  //! Mapping of recordables names to access functions
  static nest::RecordablesMap< lif_psc_alpha_sl_ip > recordablesMap_;

  /** @} */
};

inline nest::port
mynest::lif_psc_alpha_sl_ip::send_test_event( nest::Node& target,
  nest::port receptor_type,
  nest::synindex,
  bool )
{
  // You should usually not change the code in this function.
  // It confirms that the target of connection @c c accepts @c SpikeEvent on
  // the given @c receptor_type.
  nest::SpikeEvent e;
  e.set_sender( *this );
  return target.handles_test_event( e, receptor_type );
}

inline nest::port
mynest::lif_psc_alpha_sl_ip::handles_test_event( nest::SpikeEvent&,
  nest::port receptor_type )
{
  // You should usually not change the code in this function.
  // It confirms to the connection management system that we are able
  // to handle @c SpikeEvent on port 0. You need to extend the function
  // if you want to differentiate between input ports.
  if ( receptor_type != 0 )
  {
    throw nest::UnknownReceptorType( receptor_type, get_name() );
  }
  return 0;
}

inline nest::port
mynest::lif_psc_alpha_sl_ip::handles_test_event( nest::CurrentEvent&,
  nest::port receptor_type )
{
  // You should usually not change the code in this function.
  // It confirms to the connection management system that we are able
  // to handle @c CurrentEvent on port 0. You need to extend the function
  // if you want to differentiate between input ports.
  if ( receptor_type != 0 )
  {
    throw nest::UnknownReceptorType( receptor_type, get_name() );
  }
  return 0;
}

inline nest::port
mynest::lif_psc_alpha_sl_ip::handles_test_event( nest::DataLoggingRequest& dlr,
  nest::port receptor_type )
{
  // You should usually not change the code in this function.
  // It confirms to the connection management system that we are able
  // to handle @c DataLoggingRequest on port 0.
  // The function also tells the built-in UniversalDataLogger that this node
  // is recorded from and that it thus needs to collect data during simulation.
  if ( receptor_type != 0 )
  {
    throw nest::UnknownReceptorType( receptor_type, get_name() );
  }

  return B_.logger_.connect_logging_device( dlr, recordablesMap_ );
}

inline void
lif_psc_alpha_sl_ip::get_status( DictionaryDatum& d ) const
{
  // get our own parameter and state data
  P_.get( d );
  S_.get( d );

  // get information managed by parent class
  Archiving_Node::get_status( d );

  ( *d )[ nest::names::recordables ] = recordablesMap_.get_list();
}

inline void
lif_psc_alpha_sl_ip::set_status( const DictionaryDatum& d )
{
  Parameters_ ptmp = P_; // temporary copy in case of errors
  ptmp.set( d );         // throws if BadProperty
  State_ stmp = S_;      // temporary copy in case of errors
  stmp.set( d, ptmp );   // throws if BadProperty

  // We now know that (ptmp, stmp) are consistent. We do not
  // write them back to (P_, S_) before we are also sure that
  // the properties to be set in the parent class are internally
  // consistent.
  Archiving_Node::set_status( d );

  // if we get here, temporaries contain consistent set of properties
  P_ = ptmp;
  S_ = stmp;
}

} // namespace

#endif /* #ifndef LIF_PSC_ALPHA_SL_IP_H */
