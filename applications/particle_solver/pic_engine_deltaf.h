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

#include "../../src/physics/physical_constants.h"
#include "../../src/utilities/primitives.h"

namespace simpla
{

/**
 * \ingroup ParticleEngine
 * \brief \f$\delta f\f$ engine
 */

struct PICEngineDeltaF
{

private:

	Real m_;
	Real q_;

	Real cmr_, q_kT_;
public:
	static constexpr bool is_implicit = false;
	typedef PICEngineDeltaF this_type;
	typedef Vec3 coordinates_type;
	typedef Vec3 vector_type;
	typedef Real scalar_type;

	struct Point_s
	{
		coordinates_type x;
		Vec3 v;
		Real f;
		scalar_type w;

		typedef std::tuple<coordinates_type, vector_type, scalar_type, scalar_type> compact_point_s;

	};

public:
	PICEngineDeltaF(Real m = 1.0, Real q = 1.0, Real T = 1.0)
			: m_(m), q_(q), cmr_(q / m), q_kT_(1.0)
	{
		DEFINE_PHYSICAL_CONST

		q_kT_ = q / (T * boltzmann_constant);
	}

	~PICEngineDeltaF()
	{
	}

	static std::string get_type_as_string()
	{
		return "DeltaF";
	}

	Real get_mass() const
	{
		return m_;

	}
	Real get_charge() const
	{
		return q_;

	}

	template<typename TE0, typename TB0, typename TE1, typename TB1>
	inline void next_timestep(Point_s * p, Real dt, TE0 const &fE0, TB0 const & fB0, TE1 const &fE1,
	        TB1 const & fB1) const
	{
		p->x += p->v * dt * 0.5;

		vector_type B = fB1(p->x) + fB0(p->x);
		vector_type E = fE1(p->x) + fE0(p->x);

		vector_type v_;

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

	}

	template<typename TJ>
	void ScatterJ(Point_s const & p, TJ * J) const
	{
		J->scatter_cartesian(std::make_tuple(p.x, p.v), p.f * q_ * p.w);
	}

	template<typename TJ>
	void ScatterRho(Point_s const & p, TJ * rho) const
	{
		rho->scatter_cartesian(std::make_tuple(p.x, 1.0), p.f * q_ * p.w);
	}

	static inline Point_s push_forward(coordinates_type const & x, vector_type const &v, scalar_type f)
	{
		return std::move(Point_s( { x, v, f }));
	}

	static inline auto pull_back(Point_s const & p) DECL_RET_TYPE((std::make_tuple(p.x,p.v,p.f)))

};

}

} // namespace simpla

#endif /* PIC_ENGINE_DELTAF_H_ */
