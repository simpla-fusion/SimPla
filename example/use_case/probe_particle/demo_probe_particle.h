/*
 * demo_probe_particle.h
 *
 *  Created on: 2014年11月21日
 *      Author: salmon
 */

#ifndef EXAMPLE_USE_CASE_DEMO_PROBE_PARTICLE_H_
#define EXAMPLE_USE_CASE_DEMO_PROBE_PARTICLE_H_

#include "../../../core/utilities/utilities.h"
#include "../../../core/particle/particle_engine.h"

namespace simpla
{

struct ProbeDemo
{
	typedef ProbeDemo this_type;
	typedef Vec3 coordinates_type;
	typedef Vec3 vector_type;
	typedef Real scalar_type;

	SP_DEFINE_POINT_STRUCT(Point_s,
			double[3] ,x,
			double[3], v,
			Real, f,
			scalar_type, w,
			double[3][3], T
	)

	SP_DEFINE_PROPERTIES(
			Real, mass,
			Real, charge,
			Real, temperature,
			Real[3][3] , pressure
	)

	static constexpr size_t memory_length = 0; //!  declare this engine is memoryless

private:
	Real cmr_, q_kT_;
public:

	ProbeDemo() :
			mass(1.0), charge(1.0), temperature(1.0)
	{
		pressure = 0;
		update();
	}

	void update()
	{
		DEFINE_PHYSICAL_CONST

		cmr_ = charge / mass;
		q_kT_ = charge / (temperature * boltzmann_constant);
	}

	~ProbeDemo()
	{
	}

	static std::string get_type_as_string()
	{
		return "ProbeDemo";
	}

	template<typename Point_p, typename TE, typename TB>
	void next_timestep(Point_p p, Real dt, TE const &fE, TB const & fB) const
	{
//		p->x += p->v * dt * 0.5;
//
//		auto B = fB(p->x);
//		auto E = fE(p->x);
//
//		Vec3 v_;
//
//		auto t = B * (cmr_ * dt * 0.5);
//
//		p->v += E * (cmr_ * dt * 0.5);
//
//		v_ = p->v + cross(p->v, t);
//
//		v_ = cross(v_, t) / (dot(t, t) + 1.0);
//
//		p->v += v_;
//		auto a = (-dot(E, p->v) * q_kT_ * dt);
//		p->w = (-a + (1 + 0.5 * a) * p->w) / (1 - 0.5 * a);
//
//		p->v += v_;
//		p->v += E * (cmr_ * dt * 0.5);
//
//		p->x += p->v * dt * 0.5;

	}

};

}  // namespace simpla

#endif /* EXAMPLE_USE_CASE_DEMO_PROBE_PARTICLE_H_ */
