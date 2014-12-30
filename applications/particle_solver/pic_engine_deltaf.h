/*
 * pic_engine_deltaf.h
 *
 * \date  2013-12-10
 *      \author  salmon
 */

#ifndef PIC_ENGINE_DELTAF_H_
#define PIC_ENGINE_DELTAF_H_

#include <string>
#include <tuple>

#include "../../core/physics/physical_constants.h"
#include "../../core/utilities/primitives.h"
#include "../../core/utilities/ntuple.h"
#include "../../core/particle/particle_engine.h"
namespace simpla
{

/**
 * @ingroup ParticleEngine
 * \brief \f$\delta f\f$ engine
 */

template<typename Policy> class ParticleEngine;
class PolicyPICDeltaF;

typedef ParticleEngine<PolicyPICDeltaF> PICDeltaF;

template<>
struct ParticleEngine<PolicyPICDeltaF>
{
	typedef ParticleEngine<PolicyPICDeltaF> this_type;
	typedef Vec3 coordinates_type;
	typedef Vec3 vector_type;
	typedef Real scalar_type;

	SP_DEFINE_POINT_STRUCT(Point_s,
			coordinates_type ,x,
			Vec3, v,
			Real, f,
			scalar_type, w)

	SP_DEFINE_PROPERTIES(
			Real, mass,
			Real, charge,
			Real, temperature
	)

	int J_at_the_center;

private:
	Real cmr_, q_kT_;
public:

	ParticleEngine()
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

	~ParticleEngine()
	{
	}

	static std::string get_type_as_string()
	{
		return "DeltaF";
	}

	template<typename TJ, typename TE, typename TB>
	void next_timestep(Point_s * p, TJ* J, Real dt, TE const &fE, TB const & fB) const
	{
		p->x += p->v * dt * 0.5;

		auto B = fB(p->x);
		auto E = fE(p->x);

		Vec3 v_;

		auto t = B * (cmr_ * dt * 0.5);

		p->v += E * (cmr_ * dt * 0.5);

		v_ = p->v + Cross(p->v, t);

		v_ = Cross(v_, t) / (Dot(t, t) + 1.0);

		p->v += v_;
		auto a = (-Dot(E, p->v) * q_kT_ * dt);
		p->w = (-a + (1 + 0.5 * a) * p->w) / (1 - 0.5 * a);

		p->v += v_;
		p->v += E * (cmr_ * dt * 0.5);

		p->x += p->v * dt * 0.5;

		J->scatter_cartesian(std::forward_as_tuple(p->x, p->v, p->f * charge * p->w));

	}

	static inline Point_s push_forward(coordinates_type const & x, Vec3 const &v, scalar_type f)
	{
		return std::move(Point_s( { x, v, f }));
	}

	static inline auto pull_back(Point_s const & p)
	DECL_RET_TYPE((std::make_tuple(p.x,p.v,p.f)))

}
;

} // namespace simpla

#endif /* PIC_ENGINE_DELTAF_H_ */
