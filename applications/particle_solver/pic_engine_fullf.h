/*
 * pic_engine_fullf.h
 *
 * \date  2013-11-6
 *      \author  salmon
 */

#ifndef PIC_ENGINE_FULLF_H_
#define PIC_ENGINE_FULLF_H_

#include <string>
#include <tuple>

namespace simpla
{
/**
 *  \ingroup ParticleEngine
 *  \brief default PIC pusher, using Boris mover
 */
struct PICEngineFullF
{
private:

	Real m_;
	Real q_;
	Real cmr_;
public:

	static constexpr bool is_implicit = false;
	typedef PICEngineFullF this_type;
	typedef Vec3 coordinates_type;
	typedef Vec3 vector_type;
	typedef Real scalar_type;

	struct Point_s
	{
		coordinates_type x;
		vector_type v;
		scalar_type f;

		typedef std::tuple<coordinates_type, vector_type, scalar_type> compact_point_s;
	};

	PICEngineFullF(Real m = 1.0, Real q = 1.0)
			: m_(m), q_(q), cmr_(q / m)
	{
	}

	~PICEngineFullF()
	{
	}

	static std::string get_type_as_string()
	{
		return "FullF";
	}

	Real get_mass() const
	{
		return m_;

	}
	Real get_charge() const
	{
		return q_;

	}

	// x(-1/2->1/2), v(-1/2/1/2)
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

		p->v += v_ * 2.0;

		p->v += E * (cmr_ * dt * 0.5);

		p->x += p->v * dt * 0.5;

	}

	template<typename TJ>
	void ScatterJ(Point_s const & p, TJ * J) const
	{
		J->scatter_cartesian(std::make_tuple(p.x, p.v), p.f * q_);
	}

	template<typename TJ>
	void ScatterRho(Point_s const & p, TJ * rho) const
	{
		rho->scatter_cartesian(std::make_tuple(p.x, 1.0), p.f * q_);
	}

	static inline Point_s push_forward(coordinates_type const & x, vector_type const &v, scalar_type f)
	{
		return std::move(Point_s( { x, v, f }));
	}

	static inline auto pull_back(Point_s const & p) DECL_RET_TYPE((std::make_tuple(p.x,p.v,p.f)))

};

}
// namespace simpla

#endif /* PIC_ENGINE_FULLF_H_ */
