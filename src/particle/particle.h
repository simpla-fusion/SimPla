/*
 * particle.h
 *
 *  Created on: 2012-11-1
 *      Author: salmon
 */

#ifndef PARTICLE_H_
#define PARTICLE_H_

#include <fetl/primitives.h>
#include "fetl/field_rw_cache.h"
namespace simpla
{

template<typename Engine, template<typename > class PICContainer>
class Particle: public PICContainer<typename Engine::Point_s>
{

private:
	Real m_, q_;
public:
	typedef Engine engine_type;

	typedef typename engine_type::Point_s particle_type;

	typedef Particle<engine_type, PICContainer> this_type;

	typedef PICContainer<typename Engine::Point_s> base_type;

	template<typename ...Args>
	Particle(Real m, Real q, Args ... args) :
			base_type(args...), m_(m), q_(q)
	{
	}

	template<typename TP>
	inline void SetProperties(TP const &p)
	{
		m_ = p.Get<Real>("Mass");
		q_ = p.Get<Real>("Charge");
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
		base_type::ForAllParticle<void(particle_type &, Real, Real, Args...)>(
				fun, m_, q_, std::forward<Args>(args) ...);
	}

	template<typename TFUN, typename TJ, typename ... Args>
	inline void ForEach(TFUN const & fun, TJ & J, Args const & ... args) const
	{
		base_type::ForAllParticle<
				void(particle_type const &, Real, Real, Args...)>(fun, m_, q_,
				J, std::forward<Args>(args) ...);
	}

	template<typename ... Args>
	inline void Push(Args const& ... args)
	{
		base_type::ForAllParticle<void(particle_type &, Real, Real, Args...)>(
				engine_type::Push, m_, q_, std::forward<Args>(args) ...);
	}

	template<typename TFUN, typename TJ, typename ... Args>
	inline void Scatter(TFUN const & fun, TJ & J, Args const & ... args) const
	{
		base_type::ForAllParticle<
				void(particle_type const &, Real, Real, Args...)>(fun, m_, q_,
				J, std::forward<Args>(args) ...);
	}

	template<typename TJ, typename ... Args>
	inline void ScatterJ(TJ & J, Args const & ... args) const
	{
		base_type::ForAllParticle<
				void(particle_type const &, Real, Real, Args...)>(
				engine_type::ScatterJ, m_, q_, J, std::forward<Args>(args) ...);
	}

	template<typename TN, typename ... Args>
	inline void ScatterN(TN & n, Args & ... args) const
	{
		base_type::ForAllParticle<
				void(particle_type const &, Real, Real, Args...)>(
				engine_type::ScatterN, m_, q_, n, std::forward<Args>(args) ...);
	}

};

}
// namespace simpla

#endif /* PARTICLE_H_ */
