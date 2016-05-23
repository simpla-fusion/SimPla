/*
 * pic_engine_implicit.h
 *
 * @date  2014-4-10
 *      @author  salmon
 */

#ifndef PIC_ENGINE_IMPLICIT_H_
#define PIC_ENGINE_IMPLICIT_H_

#include <sstream>
#include <string>
#include <type_traits>

#include "../../src/fetl/fetl.h"
#include "../../src/physics/PhysicalConstants.h"
#include "../../src/utilities/type_traits.h"

namespace simpla
{

template<typename TM, typename Policy> class Interpolator;

/**
 *  @ingroup ParticleEngine
 *  \brief default PIC pusher, using Boris mover
 */

template<typename TM, typename TInterpolator = Interpolator<TM, std::nullptr_t>> struct PICEngineImplicit
{
public:
	enum
	{
		is_implicit = true
	};
	Real m;
	Real q;

	typedef TM mesh_type;
	typedef TInterpolator interpolator_type;
	typedef PICEngineImplicit<mesh_type, interpolator_type> this_type;

	typedef typename mesh_type::coordinate_tuple coordinate_tuple;
	typedef Vec3 vector_type;
	typedef typename mesh_type::scalar_type scalar_type;

	typedef typename mesh_type:: template field<VERTEX, scalar_type> rho_type;

	typedef typename mesh_type:: template field<VERTEX, nTuple<3, scalar_type>> J_type;

	typedef typename mesh_type:: template field<VERTEX, nTuple<3, Real> > E0_type;

	typedef typename mesh_type:: template field<VERTEX, nTuple<3, Real> > B0_type;

	typedef typename mesh_type:: template field<VERTEX, nTuple<3, scalar_type>> E1_type;

	typedef typename mesh_type:: template field<VERTEX, nTuple<3, scalar_type>> B1_type;

	struct Point_s
	{
		coordinate_tuple x;
		Vec3 v;
		Real f;

		typedef std::tuple<coordinate_tuple, Vec3, Real> compact_type;
		static DataType create_datadesc()
		{
			auto d_type = DataType::create<Point_s>();

			d_type.push_back<coordinate_tuple>("x", offsetof(Point_s, x));
			d_type.push_back<vector_type>("v", offsetof(Point_s, v));
			d_type.push_back<scalar_type>("f", offsetof(Point_s, f));

			return std::move(d_type);
		}
	};

	Real cmr_;
public:
	mesh_type const &mesh;

	PICEngineImplicit(mesh_type const &m) :
			mesh(m), m(1.0), q(1.0), cmr_(1.0)
	{
	}
	template<typename ...Others>
	PICEngineImplicit(mesh_type const &pmesh, Others && ...others) :
			PICEngineImplicit(pmesh)
	{
		load(std::forward<Others >(others)...);
	}
	template<typename TDict, typename ...Args>
	void load(TDict const& dict, Args const & ...args)
	{
		m = (dict["Mass"].template as<Real>(1.0));
		q = (dict["Charge"].template as<Real>(1.0));

		cmr_ = (q / m);

	}

	~PICEngineImplicit()
	{
	}

	static std::string get_type_as_string()
	{
		return "Implicit";
	}

	Real get_mass() const
	{
		return m;

	}
	Real get_charge() const
	{
		return q;

	}
	template<typename OS>
	OS & print(OS & os) const
	{

		DEFINE_PHYSICAL_CONST
		;

		os << "Engine = '" << get_type_as_string() << "' "

		<< " , " << "Mass = " << m / proton_mass << " * m_p"

		<< " , " << "Charge = " << q / elementary_charge << " * q_e"

		;
		return os;

	}

	template<typename TE0, typename TE1, typename TB0, typename TB1, typename TJ>
	inline void next_timestep(Point_s * p, Real dt, E0_type const &fE0, B0_type const & fB0, E1_type const &fE1,
	        B1_type const & fB1, TJ *J) const
	{

		auto B = interpolator_type::gather_cartesian(fB0, p->x) + real(interpolator_type::gather_cartesian(fB1, p->x));

		auto E = interpolator_type::gather_cartesian(fE0, p->x) + real(interpolator_type::gather_cartesian(fE1, p->x));

		Vec3 v_;

		auto t = B * (cmr_ * dt * 0.5);

		p->v += E * (cmr_ * dt * 0.5);

		v_ = p->v + Cross(p->v, t);

		v_ = Cross(v_, t) / (Dot(t, t) + 1.0);

		p->v += v_ * 2.0;

		p->v += E * (cmr_ * dt * 0.5);

		p->x += p->v * dt;

		interpolator_type::scatter_cartesian(J, std::make_tuple(p->x, p->v), p->f * q);

	}

	void Scatter(Point_s const & p, J_type * J) const
	{
		interpolator_type::scatter_cartesian(J, std::make_tuple(p.x, p.v), p.f * q);
	}

	void Scatter(Point_s const & p, rho_type * n) const
	{
		interpolator_type::scatter_cartesian(n, std::make_tuple(p.x, 1.0), p.f * q);
	}

	static inline Point_s make_point(coordinate_tuple const & x, Vec3 const &v, Real f)
	{
		return std::move(Point_s( { x, v, f }));
	}

};
template<typename OS, typename ... TM> OS&
operator<<(OS& os, typename PICEngineImplicit<TM...>::Point_s const & p)
{
	os << "{ x= {" << p.x << "} , v={" << p.v << "}, f=" << p.f << " }";

	return os;
}
}
// namespace simpla

#endif /* PIC_ENGINE_IMPLICIT_H_ */
