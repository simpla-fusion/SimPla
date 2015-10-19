/*
 * demo_pic.h
 *
 *  Created on: 2014-11-21
 *      Author: salmon
 */

#ifndef EXAMPLE_USE_CASE_PIC_DEMO_PIC_H_
#define EXAMPLE_USE_CASE_PIC_DEMO_PIC_H_

#include <algorithm>
#include <string>
#include <tuple>

#include "../../core/dataset/datatype.h"
#include "../../core/dataset/datatype_ext.h"

#include "../../core/gtl/ntuple.h"
#include "../../core/gtl/primitives.h"
#include "../../core/gtl/type_traits.h"
#include "../../core/particle/particle_engine.h"
#include "../../core/physics/physical_constants.h"

#include "../../core/manifold/domain.h"

using namespace simpla;

namespace simpla
{
SP_DEFINE_STRUCT(pic_demo,
		Vec3, x,
		Vec3, v,
		Real, f,
		Real, w)

template<typename TBase>
struct FiberBundle<pic_demo, TBase>
{
	typedef FiberBundle<pic_demo, TBase> this_type;
public:
	typedef Vec3 vector_type;
	typedef Real scalar_type;


	typedef pic_demo point_type;


	typedef TBase base_manifold;

	typedef TBase::range_type range_type;

	typedef Domain<base_manifold> domain_type;


private:
	base_manifold const &m_mesh_;
	domain_type m_domain_;

	SP_DEFINE_PROPERTIES(
			Real, mass,
			Real, charge,
			Real, temperature
	)

private:
	Real m_cmr_, m_q_kT_;
public:

	FiberBundle() :
			m_mass(1.0), m_charge(1.0), m_temperature(1.0)
	{
		update();
	}

	void update()
	{
		DEFINE_PHYSICAL_CONST
		m_cmr_ = m_charge / m_mass;
		m_q_kT_ = m_charge / (m_temperature * boltzmann_constant);
	}

	~FiberBundle()
	{
	}

	template<typename ...Args>
	static inline typename base_manifold::point_type project(point_type const &p, Args &&...args)
	{
		return p.x;
	}

	template<typename TV, typename ...Args>
	static inline point_type lift(typename base_manifold::point_type const &x, TV const &v, Real f, Args &&...args)
	{
		point_type res{x, v, f};

		return std::move(res);
	}

	template<typename ...Args>
	static inline vector_type push_forward(point_type const &p, Args &&...args)
	{
		vector_type res;
		res = p.v * p.f;
		return std::move(res);
	}


	template<typename TE, typename TB, typename TJ>
	void next_time_step(point_type *p0, Real dt, TE const &fE, TB const &fB,
			TJ *J)
	{
		p0->x += p0->v * dt * 0.5;

		auto B = pull_back(fB, project(*p0));
		auto E = pull_back(fE, project(*p0));

		Vec3 v_;

		auto t = B * (m_cmr_ * dt * 0.5);

		p0->v += E * (m_cmr_ * dt * 0.5);

		v_ = p0->v + cross(p0->v, t);

		v_ = cross(v_, t) / (inner_product(t, t) + 1.0);

		p0->v += v_;
		auto a = (-inner_product(E, p0->v) * m_q_kT_ * dt);
		p0->w = (-a + (1 + 0.5 * a) * p0->w) / (1 - 0.5 * a);

		p0->v += v_;
		p0->v += E * (m_cmr_ * dt * 0.5);

		p0->x += p0->v * dt * 0.5;

		m_mesh_.scatter(J, project(*p0), push_forward(*p0));

	}

};


}  // namespace simpla

#endif /* EXAMPLE_USE_CASE_PIC_DEMO_PIC_H_ */
