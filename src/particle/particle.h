/*
 * particle.h
 *
 *  Created on: 2012-11-1
 *      Author: salmon
 */

#ifndef PARTICLE_H_
#define PARTICLE_H_

#include <fetl/primitives.h>
#include <particle/pic.h>
#include <cstddef>

namespace simpla
{

template<template<typename > class Engine, typename TM>
class Particle: public PIC<TM,
		typename Engine<typename TM::coordinates_type>::Point_s>
{

private:
	Real m_, q_;
public:
	typedef Engine<typename TM::coordinates_type> engine_type;

	typedef typename engine_type::Point_s value_type;

	typedef typename engine_type::Point_s particle_type;

	typedef Particle<Engine, TM> this_type;

	typedef PIC<TM, typename Engine<typename TM::coordinates_type>::Point_s> base_type;

	template<typename ...Args>
	Particle(Real m, Real q, Args ... args) :
			base_type(args...), m_(m), q_(q)
	{
	}

	template<typename TP>
	inline void SetProperties(TP const &p)
	{
		m_ = p.template Get<Real>("Mass");
		q_ = p.template Get<Real>("Charge");
	}

	template<typename TP>
	inline void GetProperties(TP &p) const
	{
		p.Set("Mass", m_);
		p.Set("Charge", q_);
	}

	void Init(size_t num_pic)
	{
		value_type default_value;

		engine_type::SetDefaultValue(default_value);

		base_type::ResizeCells(num_pic, default_value);
	}

	template<typename TFUN, typename ... Args>
	inline void ForEach(TFUN const & fun, Args const& ... args)
	{
		base_type::template ForAllParticle<
				void(particle_type &, Real, Real, Args...)>(fun, m_, q_,
				std::forward<Args>(args) ...);
	}

	template<typename TFUN, typename TJ, typename ... Args>
	inline void ForEach(TFUN const & fun, TJ & J, Args const & ... args) const
	{
		base_type::template ForAllParticle<
				void(particle_type const &, Real, Real, Args...)>(fun, m_, q_,
				J, args ...);
	}

	template<typename ... Args>
	inline void Push(Args const& ... args)
	{
		base_type::template ForAllParticle<
				void(particle_type&, Real, Real, Args const &...)>(
				engine_type::Push, m_, q_, args ...);
	}

	template<typename TFUN, typename TJ, typename ... Args>
	inline void Scatter(TFUN const & fun, TJ & J, Args const & ... args) const
	{
		base_type::template ForAllParticle<
				void(particle_type&, Real, Real, TJ&, Args const &...)>(fun, m_,
				q_, J, args ...);
	}

	template<typename TJ, typename ... Args>
	inline void ScatterJ(TJ & J, Args const & ... args) const
	{
		base_type::template ForAllParticle<
				void(particle_type&, Real, Real, TJ &, Args const &...)>(
				engine_type::ScatterJ, m_, q_, J, args ...);
	}

	template<typename TN, typename ... Args>
	inline void ScatterN(TN & n, Args & ... args) const
	{
		base_type::template ForAllParticle<
				void(particle_type&, Real, Real, TN &, Args const &...)>(
				engine_type::ScatterJ, m_, q_, n, args ...);
	}

};

}
// namespace simpla

#endif /* PARTICLE_H_ */
