/**
 * @file simple_particle.h
 *
 * @date 2015-2-13
 * @author salmon
 */

#ifndef CORE_PARTICLE_SIMPLE_PARTICLE_H_
#define CORE_PARTICLE_SIMPLE_PARTICLE_H_

#include "particle_engine.h"
#include "simple_particle_generator.h"

namespace simpla
{
struct SimpleParticleEngine
{
	typedef double scalar_type;
	typedef nTuple<scalar_type, 3> coordinate_tuple;
	typedef nTuple<scalar_type, 3> vector_type;

	SP_DEFINE_POINT_STRUCT(Point_s,
			coordinate_tuple,x,
			vector_type, v,
			scalar_type, f )

	SP_DEFINE_PROPERTIES(
			scalar_type, mass,
			scalar_type, charge,
			scalar_type, temperature
	)

private:
	Real cmr_, q_kT_;
public:

	SimpleParticleEngine()
			: mass(1.0), charge(1.0), temperature(1.0)
	{
		update();
	}

	void update()
	{
		DEFINE_PHYSICAL_CONST
		cmr_ = charge / mass;
		q_kT_ = charge / (temperature * boltzmann_constant);
	}

	~SimpleParticleEngine()
	{
	}

	static std::string get_type_as_string()
	{
		return "DeltaF";
	}

	template<typename TE, typename TB, typename TJ>
	void next_timestep(Point_s * p, Real dt, TE const &fE, TB const & fB,
			TJ * J) const
	{
		p->x += p->v * dt * 0.5;

		auto B = fB(p->x);
		auto E = fE(p->x);

		Vec3 v_;

		auto t = B * (cmr_ * dt * 0.5);

		p->v += p->v + E * (cmr_ * dt * 0.5);

		v_ = p->v + Cross(p->v, t);

		v_ = Cross(v_, t) / (Dot(t, t) + 1.0) * 2;

		p->v += v_;

		p->v += E * (cmr_ * dt * 0.5);

		p->x += p->v * dt * 0.5;

		J->scatter_cartesian(std::forward_as_tuple(p->x, p->v, p->f * charge));

	}

	Point_s push_forward(coordinate_tuple const & x, vector_type const &v,
			Real f = 1.0) const
	{
		return Point_s( { x, v, f });
	}

};

}
// namespace simpla

#endif /* CORE_PARTICLE_SIMPLE_PARTICLE_H_ */
