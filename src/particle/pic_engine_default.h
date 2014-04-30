/*
 * pic_engine_default.h
 *
 *  Created on: 2013年11月6日
 *      Author: salmon
 */

#ifndef PIC_ENGINE_DEFAULT_H_
#define PIC_ENGINE_DEFAULT_H_

#include <cstddef>
#include <sstream>
#include <string>

#include "../fetl/ntuple.h"
#include "../fetl/primitives.h"
#include "../physics/physical_constants.h"
#include "particle.h"

namespace simpla
{

template<typename TM, bool IsImplicit, typename Interpolator = typename TM::interpolator_type>
struct PICEngineDefault
{
public:
	enum
	{
		EnableImplicit = IsImplicit
	};
	const Real m;
	const Real q;

	typedef PICEngineDefault<TM, IsImplicit, Interpolator> this_type;
	typedef TM mesh_type;
	typedef Interpolator interpolator_type;

	typedef typename mesh_type::coordinates_type coordinates_type;
	typedef typename mesh_type::scalar_type scalar_type;

	typedef Field<mesh_type, VERTEX, scalar_type> n_type;

	typedef typename std::conditional<EnableImplicit, Field<mesh_type, VERTEX, nTuple<3, scalar_type>>,
	        Field<mesh_type, EDGE, scalar_type> >::type J_type;

	typedef nTuple<7, Real> storage_value_type;

	struct Point_s
	{
		coordinates_type x;
		Vec3 v;
		Real f;

		static std::string DataTypeDesc()
		{
			std::ostringstream os;
			os

			<< "H5T_COMPOUND {          "

			<< "   H5T_ARRAY { [3] H5T_NATIVE_DOUBLE}    \"x\" : " << (offsetof(Point_s, x)) << ";"

			<< "   H5T_ARRAY { [3] H5T_NATIVE_DOUBLE}    \"v\" :  " << (offsetof(Point_s, v)) << ";"

			<< "   H5T_NATIVE_DOUBLE    \"f\" : " << (offsetof(Point_s, f)) << ";"

			<< "}";

			return os.str();
		}

	};

private:
	Real cmr_;
public:
	mesh_type const &mesh;

public:

	template<typename TDict, typename ...Args>
	PICEngineDefault(mesh_type const &pmesh, TDict const & dict, Args const & ...args)
			: mesh(pmesh),

			m(dict["Mass"].template as<Real>(1.0)),

			q(dict["Charge"].template as<Real>(1.0)),

			cmr_(q / m)
	{
	}

	~PICEngineDefault()
	{
	}

	static std::string GetTypeAsString()
	{
		return "Default";
	}

	std::string Dump(std::string const & path = "", bool is_verbose = false) const
	{
		std::stringstream os;

		DEFINE_PHYSICAL_CONST(mesh.constants());

		os << "Engine = '" << GetTypeAsString() << "' "

		<< " , " << "Mass = " << m / proton_mass << " * m_p"

		<< " , " << "Charge = " << q / elementary_charge << " * q_e"

		;
		return os.str();

	}
	static inline Point_s DefaultValue()
	{
		Point_s p;
		p.f = 1.0;
		return std::move(p);
	}

	template<typename TJ, typename TE, typename TB, typename ... Others>
	inline void NextTimeStepZero(Point_s * p, Real dt, TJ *J, TE const &fE, TB const & fB,
	        Others const &...others) const
	{
		NextTimeStepZero(Bool2Type<EnableImplicit>(), p, dt, J, fE, fB);
	}
	template<typename TE, typename TB, typename ... Others>
	inline void NextTimeStepHalf(Point_s * p, Real dt, TE const &fE, TB const & fB, Others const &...others) const
	{
		NextTimeStepHalf(Bool2Type<EnableImplicit>(), p, dt, fE, fB);
	}
	// x(-1/2->1/2),v(0)
	template<typename TE, typename TB, typename ... Others>
	inline void NextTimeStepZero(Bool2Type<true>, Point_s * p, Real dt,
	        Field<mesh_type, VERTEX, nTuple<3, scalar_type> > *J, TE const &fE, TB const & fB,
	        Others const &...others) const
	{
		//		auto B = interpolator_type::Gather(fB, p->x);
		//		auto E = interpolator_type::Gather(fE, p->x);

		p->x += p->v * dt;
		Vec3 v;
		v = p->v * p->f * q;
		interpolator_type::Scatter(p->x, v, J);
	}
	// v(0->1)
	template<typename TE, typename TB, typename ... Others>
	inline void NextTimeStepHalf(Bool2Type<true>, Point_s * p, Real dt, TE const &fE, TB const & fB,
	        Others const &...others) const
	{

		auto B = interpolator_type::Gather(fB, p->x);
		auto E = interpolator_type::Gather(fE, p->x);

		Vec3 v_;

		auto t = B * (cmr_ * dt * 0.5);

		p->v += E * (cmr_ * dt * 0.5);

		v_ = p->v + Cross(p->v, t);

		v_ = Cross(v_, t) / (Dot(t, t) + 1.0);

		p->v += v_ * 2.0;

		p->v += E * (cmr_ * dt * 0.5);

	}
	// x(-1/2->1/2), v(-1/2/1/2)
	template<typename TJ, typename TE, typename TB, typename ... Others>
	inline void NextTimeStepZero(Bool2Type<false>, Point_s * p, Real dt, TJ *J, TE const &fE, TB const & fB,
	        Others const &...others) const
	{

		p->x += p->v * dt * 0.5;
		auto B = interpolator_type::Gather(fB, p->x);
		auto E = interpolator_type::Gather(fE, p->x);

		Vec3 v_;

		auto t = B * (cmr_ * dt * 0.5);

		p->v += E * (cmr_ * dt * 0.5);

		v_ = p->v + Cross(p->v, t);

		v_ = Cross(v_, t) / (Dot(t, t) + 1.0);

		p->v += v_ * 2.0;

		p->v += E * (cmr_ * dt * 0.5);

		p->x += p->v * dt * 0.5;
		Vec3 v;
		v = p->v * p->f * q;
		interpolator_type::Scatter(p->x, v, J);

	}
	template<typename ... Others>
	inline void NextTimeStepHalf(Bool2Type<false>, Point_s * p, Real dt, Field<mesh_type, EDGE, scalar_type> const &fE,
	        Field<mesh_type, FACE, scalar_type> const & fB, Others const &...others) const
	{
	}

	template<int IFORM, typename TV, typename ...Args>
	void Scatter(Point_s const & p, Field<mesh_type, IFORM, TV> * n, Args const & ...) const
	{
		interpolator_type::Scatter(p.x, q * p.f, n);
	}

	static inline Point_s make_point(coordinates_type const & x, Vec3 const &v, Real f)
	{
		return std::move(Point_s( { x, v, f }));
	}

};

template<typename OS, typename ... TM> OS&
operator<<(OS& os, typename PICEngineDefault<TM...>::Point_s const & p)
{
	os << "{ x= {" << p.x << "} , v={" << p.v << "}, f=" << p.f << " }";

	return os;
}

} // namespace simpla

#endif /* PIC_ENGINE_DEFAULT_H_ */
