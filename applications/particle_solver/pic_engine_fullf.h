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
#include "../../src/utilities/data_type.h"
#include "../../src/utilities/primitives.h"

namespace simpla
{
/**
 *  @ingroup ParticleEngine
 *  \brief default PIC pusher, using Boris mover
 */

class PICEngineFullF
{

	Real m_;
	Real q_;
	Real cmr_;

public:

	typedef PICEngineFullF this_type;
	typedef Vec3 coordinate_tuple;
	typedef Vec3 vector_type;
	typedef Real scalar_type;

	struct Point_s
	{
		coordinate_tuple x;
		vector_type v;
		scalar_type f;

		typedef std::tuple<coordinate_tuple, vector_type, scalar_type> compact_type;

		static DataType create_datadesc()
		{
			auto d_type = DataType::create<Point_s>();

			d_type.push_back<coordinate_tuple>("x", offsetof(Point_s, x));
			d_type.push_back<vector_type>("v", offsetof(Point_s, v));
			d_type.push_back<scalar_type>("f", offsetof(Point_s, f));

			return std::move(d_type);
		}

	};

	PICEngineFullF(Real m = 1.0, Real q = 1.0)
			: m_(m), q_(q), cmr_(q / m)
	{
	}

	~PICEngineFullF()
	{
	}

	template<typename TDict>
	void load(TDict const & dict)

	{
		m_ = (dict["Mass"].template as<Real>(1.0));

		q_ = (dict["Charge"].template as<Real>(1.0));

		cmr_ = (q_ / m_);

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
//
//	// x(-1/2->1/2), v(-1/2/1/2)
	template<typename TE, typename TB>
	void next_timestep(Point_s * p, Real dt, TE const &fE, TB const & fB) const
	{

		p->x += p->v * dt * 0.5;

		auto B = fB(p->x);
		auto E = fE(p->x);

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

	static inline Point_s push_forward(coordinate_tuple const & x, vector_type const &v, scalar_type f)
	{
		return std::move(Point_s( { x, v, f }));
	}

	static inline std::tuple<coordinate_tuple, vector_type, scalar_type> pull_back(Point_s const & p)
	{
		return ((std::make_tuple(p.x, p.v, p.f)));
	}

}
;

}
// namespace simpla

#endif /* PIC_ENGINE_FULLF_H_ */
