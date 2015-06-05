/*
 * demo_pic.h
 *
 *  Created on: 2014年11月21日
 *      Author: salmon
 */

#ifndef EXAMPLE_USE_CASE_PIC_DEMO_PIC_H_
#define EXAMPLE_USE_CASE_PIC_DEMO_PIC_H_

#include <algorithm>
#include <string>
#include <tuple>

#include "../../core/dataset/datatype.h"
#include "../../core/gtl/ntuple.h"
#include "../../core/gtl/primitives.h"
#include "../../core/gtl/type_traits.h"
#include "../../core/particle/particle_engine.h"
#include "../../core/physics/physical_constants.h"

using namespace simpla;

namespace simpla
{

struct PICDemo
{
	typedef PICDemo this_type;
	typedef Vec3 point_type;
	typedef Vec3 vector_type;
	typedef Real scalar_type;

	SP_DEFINE_STRUCT(Point_s,
			point_type ,x,
			Vec3, v,
			Real, f,
			Real, w)

	SP_DEFINE_PROPERTIES(
			Real, mass,
			Real, charge,
			Real, temperature
	)

private:
	Real m_cmr_, m_q_kT_;
public:

	PICDemo() :
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

	~PICDemo()
	{
	}

	static std::string get_type_as_string()
	{
		return "PICDemo";
	}

	template<typename TE, typename TB, typename TJ>
	void next_timestep(Point_s * p0, Real dt, TE const &fE, TB const & fB,
			TJ * J)
	{
		p0->x += p0->v * dt * 0.5;

		auto B = fB(p0->x);
		auto E = fE(p0->x);

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

		J->scatter(p0->x, p0->v, p0->f);

	}

	static inline Point_s push_forward(point_type const & x,
			Vec3 const &v, Real f = 1.0)
	{
		return std::move(Point_s({ x, v, f }));
	}
	static inline void push_forward(point_type const & x, Vec3 const &v,
			Real f, Point_s * p)
	{
		p->x = x;
		p->v = v;
		p->f = f;
	}
	static inline auto pull_back(Point_s const & p)
	DECL_RET_TYPE((std::make_tuple(p.x,p.v,p.f )))
};

}  // namespace simpla

#endif /* EXAMPLE_USE_CASE_PIC_DEMO_PIC_H_ */
