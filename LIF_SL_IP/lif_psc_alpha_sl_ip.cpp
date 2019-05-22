/*
 *  lif_psc_alpha_sl_ip.cpp
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

#include "lif_psc_alpha_sl_ip.h"

// C++ includes:
#include <limits>
#include <vector> //*

// Includes from libnestutil:
#include "numerics.h"

// Includes from nestkernel:
#include "exceptions.h"
#include "kernel_manager.h"
#include "universal_data_logger_impl.h"
#include "common_synapse_properties.h"//*_
#include "connection.h"
#include "connector_base.h"
#include "connector_model.h"
#include "event.h"
// Includes from sli:
#include "dict.h"
#include "dictutils.h"
#include "doubledatum.h"
#include "integerdatum.h"
#include "lockptrdatum.h"

#define mostrar(contenido) cout << "Resultado  " #contenido "==" << ":" << contenido << endl;

using namespace nest;

/* ----------------------------------------------------------------
 * Recordables map
 * ---------------------------------------------------------------- */

nest::RecordablesMap< mynest::lif_psc_alpha_sl_ip >
  mynest::lif_psc_alpha_sl_ip::recordablesMap_;

namespace nest
{
// Override the create() method with one call to RecordablesMap::insert_()
// for each quantity to be recorded.
template <>
void
RecordablesMap< mynest::lif_psc_alpha_sl_ip >::create()
{
  // use standard names whereever you can for consistency!
  insert_( names::V_m, &mynest::lif_psc_alpha_sl_ip::get_V_m_ );
  insert_( names::soma_exc, &mynest::lif_psc_alpha_sl_ip::get_soma_exc_ );
}
}

/* ----------------------------------------------------------------
 * Default constructors defining default parameters and state
 * ---------------------------------------------------------------- */

mynest::lif_psc_alpha_sl_ip::Parameters_::Parameters_()
  : C_m( 250.0 )     // pF
  , I_e( 0.0 )       // nA
  , tau_syn( 2.0 )   // ms
  , V_th( 35.0 )    // mV
  , V_reset( -70.0 ) // mV
  , t_ref( 2.0 )     // ms
  , Decay( 0.02 )    // mV Decay parameter
  , dt( 0.0 )    // ms Parameter of time resolution
  , neuromodulator() // GID of neuromodulator
  , lambda(0.0001) // Lambda - Intrinsic Plasticity enhance
  , tau(12.5) // Tau - Intrinsic Plasticity window
  , std_mod(true) // ON/OFF of the spike time dependent modification
{
}

mynest::lif_psc_alpha_sl_ip::State_::State_( const Parameters_& p )
  : V_m( p.V_reset )
  , dI_syn( 0.0 )
  , I_syn( 0.0 )
  , I_ext( 0.0 )
  , refr_count( 0 )
  , enhancement(1.0)
  , t_lastspike_(-1.0)
{
}

/* ----------------------------------------------------------------
 * Parameter and state extractions and manipulation functions
 * ---------------------------------------------------------------- */

void
mynest::lif_psc_alpha_sl_ip::Parameters_::get( DictionaryDatum& d ) const
{
  ( *d )[ names::C_m ] = C_m;
  ( *d )[ names::I_e ] = I_e;
  ( *d )[ names::tau_syn ] = tau_syn;
  ( *d )[ names::V_th ] = V_th;
  ( *d )[ names::V_reset ] = V_reset;
  ( *d )[ names::t_ref ] = t_ref;
  def< double >( d, names::tau_decay, Decay );
  def< double >( d, names::lambda, lambda );
  def< double >( d, names::tau, tau );
  def< bool >(d, names::std_mod, std_mod );
  ArrayDatum neuromodulator;
  for ( size_t j = 0; j < neuromodulator.size(); ++j )
  {
    neuromodulator.push_back( neuromodulator[ j ] );
  }
  def< ArrayDatum >( d, names::stimulator, neuromodulator );
  ArrayDatum prev_w;
  for ( size_t j = 0; j < prev_w.size(); ++j )
  {
    prev_w.push_back( neuromodulator[ j ] );
  }
}

void
mynest::lif_psc_alpha_sl_ip::Parameters_::set( const DictionaryDatum& d )
{
  updateValue< double >( d, names::C_m, C_m );
  updateValue< double >( d, names::I_e, I_e );
  updateValue< double >( d, names::tau_syn, tau_syn );
  updateValue< double >( d, names::V_th, V_th );
  updateValue< double >( d, names::V_reset, V_reset );
  updateValue< double >( d, names::t_ref, t_ref );
  updateValue< double >( d, names::tau_decay, Decay );
  updateValue< double >( d, names::lambda, lambda );
  updateValue< double >( d, names::tau, tau );
  updateValue< std::vector< long > >( d, names::stimulator, neuromodulator );
  updateValue< bool >(d,names::std_mod, std_mod );
  if ( C_m <= 0 )
  {
    throw nest::BadProperty(
      "The membrane capacitance must be strictly positive." );
  }
  if ( tau_syn <= 0 )
  {
    throw nest::BadProperty(
      "The synaptic time constant must be strictly positive." );
  }
  if ( V_reset >= V_th )
  {
    throw nest::BadProperty( "The reset potential must be below threshold." );
  }
  if ( t_ref < 0 )
  {
    throw nest::BadProperty(
      "The refractory time must be at least one simulation step." );
  }
}

void
mynest::lif_psc_alpha_sl_ip::State_::get( DictionaryDatum& d ) const
{
  // Only the membrane potential is shown in the status; one could show also the
  // other
  // state variables
  ( *d )[ names::V_m ] = V_m;
}

void
mynest::lif_psc_alpha_sl_ip::State_::set( const DictionaryDatum& d,
  const Parameters_& p )
{
  // Only the membrane potential can be set; one could also make other state
  // variables
  // settable.
  updateValue< double >( d, names::V_m, V_m );
}

mynest::lif_psc_alpha_sl_ip::Buffers_::Buffers_( lif_psc_alpha_sl_ip& n )
  : logger_( n )
{
}

mynest::lif_psc_alpha_sl_ip::Buffers_::Buffers_( const Buffers_&, lif_psc_alpha_sl_ip& n )
  : logger_( n )
{
}


/* ----------------------------------------------------------------
 * Default and copy constructor for node
 * ---------------------------------------------------------------- */

mynest::lif_psc_alpha_sl_ip::lif_psc_alpha_sl_ip()
  : Archiving_Node()
  , P_()
  , S_( P_ )
  , B_( *this )
{
  recordablesMap_.create();
}

mynest::lif_psc_alpha_sl_ip::lif_psc_alpha_sl_ip( const lif_psc_alpha_sl_ip& n )
  : Archiving_Node( n )
  , P_( n.P_ )
  , S_( n.S_ )
  , B_( n.B_, *this )
{
}

/* ----------------------------------------------------------------
 * Node initialization functions
 * ---------------------------------------------------------------- */

void
mynest::lif_psc_alpha_sl_ip::init_state_( const Node& proto )
{
  const lif_psc_alpha_sl_ip& pr = downcast< lif_psc_alpha_sl_ip >( proto );
  S_ = pr.S_;
}

void
mynest::lif_psc_alpha_sl_ip::init_buffers_()
{
  B_.spikes.clear();   // includes resize
  B_.currents.clear(); // include resize
  B_.logger_.reset();  // includes resize
}

void
mynest::lif_psc_alpha_sl_ip::calibrate()
{
  B_.logger_.init();

  const double h = Time::get_resolution().get_ms();
  P_.dt = h;
  const double eh = std::exp( -h / P_.tau_syn );
  const double tc = P_.tau_syn / P_.C_m;

  // compute matrix elements, all other elements 0
  V_.P11 = eh;
  V_.P22 = eh;
  V_.P21 = h * eh;
  V_.P30 = h / P_.C_m;
  V_.P31 = tc * ( P_.tau_syn - ( h + P_.tau_syn ) * eh );
  V_.P32 = tc * ( 1 - eh );
  // P33_ is 1

  // initial value ensure normalization to max amplitude 1.0
  V_.pscInitialValue = 1.0 * numerics::e / P_.tau_syn;

  // refractory time in steps
  V_.t_ref_steps = Time( Time::ms( P_.t_ref ) ).get_steps();
  assert(
    V_.t_ref_steps >= 0 ); // since t_ref_ >= 0, this can only fail in error
}

/* ----------------------------------------------------------------
 * Update and spike handling functions
 * ---------------------------------------------------------------- */

void
mynest::lif_psc_alpha_sl_ip::update( Time const& slice_origin,
  const long from_step,
  const long to_step )
{
  for ( long lag = from_step; lag < to_step; ++lag )
  {
    // order is important in this loop, since we have to use the old values
    // (those upon entry to the loop) on right hand sides everywhere
    
    // check for top-threshold crossing
    if ( S_.V_m == 35.0 )
    {
      // reset neuron
      S_.refr_count = V_.t_ref_steps;
      // send spike, and set spike time in archive.
      //set_spiketime( Time::step( slice_origin.get_steps() + lag + 1 ) );
      //SpikeEvent se;
     // kernel().event_delivery_manager.send( *this, se, lag );
    }
    
    // update membrane potential
    if ( S_.refr_count == 0 ) // neuron absolute not refractory
    {
      if (S_.V_m < -54.4)
      {
      	S_.V_m += ((V_.P30 * ( S_.I_ext + P_.I_e ) + V_.P31 * S_.dI_syn + V_.P32 * S_.I_syn))*S_.enhancement -(P_.Decay*(S_.V_m + 70)) * P_.dt; 
      	
      	if (S_.V_m < P_.V_reset)
      	{
      	  S_.V_m = P_.V_reset;
      	}
      }
      else
      {
      	
      	S_.V_m += (V_.P30 * ( S_.I_ext + P_.I_e ) + V_.P31 * S_.dI_syn + V_.P32 * S_.I_syn)*S_.enhancement;
      	
      	
      	
      	
      	
      	
      	
      	
      	      	     	
      	S_.Vpositivo = (S_.V_m+70)/15;
      	
      	S_.V_m = (S_.Vpositivo + (pow((S_.Vpositivo-1),2)*P_.dt)/(1-(S_.Vpositivo - 1)*P_.dt)) * 15 - 70;
      	
      	
      	
      	if (S_.V_m >= 35.0)//(S_.V_m >= -44.5)
      	{
      	  S_.V_m = 35.0;
      	  // send spike, and set spike time in archive.
      		set_spiketime( Time::step( slice_origin.get_steps() + lag + 1 ) );
      		SpikeEvent se;
     		kernel().event_delivery_manager.send( *this, se, lag );
      	}
      	
      	
      }       
    }
    else
    {
      S_.V_m = P_.V_reset; //
      --S_.refr_count;
    } // count down refractory time

    // update synaptic currents
    S_.I_syn = V_.P21 * S_.dI_syn + V_.P22 * S_.I_syn;
    S_.dI_syn *= V_.P11;
    
    // add synaptic input currents for this step
    S_.dI_syn += V_.pscInitialValue * B_.spikes.get_value( lag );
    
    // set new input current
    S_.I_ext = B_.currents.get_value( lag );

    // log membrane potential
    B_.logger_.record_data( slice_origin.get_steps() + lag );

	
  }
}

void
mynest::lif_psc_alpha_sl_ip::handle( SpikeEvent& e )
{
  assert( e.get_delay() > 0 );

  B_.spikes.add_value(
    e.get_rel_delivery_steps( kernel().simulation_manager.get_slice_origin() ),
    e.get_weight() );
    
    
    
    if (P_.std_mod)
    {
    
    size_t origSize = P_.neuromodulator.size();
    
    for (size_t i=0; i < origSize; i++)
    {
    long modulator = P_.neuromodulator[ i ];
    long source_gid = e.get_sender_gid();
    if (source_gid == modulator) ///////////////////////////////////////////////////////////////////// 
    {
    
     
  double t_spike = e.get_stamp().get_ms(); 
  Node* target = kernel().node_manager.get_node(e.get_receiver_gid());
  //Node* target = get_target();
  //double dendritic_delay = get_delay();
  // get spike history in relevant range (t1, t2] from post-synaptic neuron
  std::deque< histentry >::iterator start;
  std::deque< histentry >::iterator finish;
  // For a new synapse, t_lastspike_ contains the point in time of the last
  // spike. So we initially read the
  // history(t_last_spike - dendritic_delay, ..., T_spike-dendritic_delay]
  // which increases the access counter for these entries.
  // At registration, all entries' access counters of
  // history[0, ..., t_last_spike - dendritic_delay] have been
  // incremented by Archiving_Node::register_stdp_connection(). See bug #218 for
  // details.
  
  //long dendritic_delay = 0.0;
  
  target->get_history( S_.t_lastspike_,
    t_spike,
    &start,
&finish );
// facilitation due to post-synaptic spikes since last pre-synaptic spike
  while ( start != finish )
  {
  
 S_.enhancement = S_.enhancement + std::exp(((S_.t_lastspike_ + e.get_delay()) - start->t_ )/P_.tau)*P_.lambda; // *1.5

 S_.enhancement = S_.enhancement - std::exp((start->t_  - (t_spike + e.get_delay()))/P_.tau)*P_.lambda;
 
 ++start;
  }
  
 S_.t_lastspike_ = t_spike;
}
}
}
}

void
mynest::lif_psc_alpha_sl_ip::handle( CurrentEvent& e )
{
  assert( e.get_delay() > 0 );

  B_.currents.add_value(
    e.get_rel_delivery_steps( kernel().simulation_manager.get_slice_origin() ),
    e.get_weight() * e.get_current() );
}

// Do not move this function as inline to h-file. It depends on
// universal_data_logger_impl.h being included here.
void
mynest::lif_psc_alpha_sl_ip::handle( DataLoggingRequest& e )
{
  B_.logger_.handle( e ); // the logger does this for us
}
