/*
 * pic_engine_deltaf.h
 *
 *  Created on: 2013年12月10日
 *      Author: salmon
 */

#ifndef PIC_ENGINE_DELTAF_H_
#define PIC_ENGINE_DELTAF_H_

#include <cstddef>
#include <sstream>
#include <string>

#include "../../src/fetl/ntuple.h"
#include "../../src/fetl/primitives.h"
#include "../../src/physics/physical_constants.h"

namespace simpla
{

template<typename TM, bool IsImplicit, typename Interpolator = typename TM::interpolator_type>
struct PICEngineDeltaF
{

public:
	enum
	{
		EnableImplicit = IsImplicit
	};
	const Real m;
	const Real q;

	typedef PICEngineDeltaF<TM, IsImplicit, Interpolator> this_type;
	typedef TM mesh_type;
	typedef Interpolator interpolator_type;

	typedef typename mesh_type::coordinates_type coordinates_type;
	typedef typename mesh_type::scalar_type scalar_type;

	typedef Field<mesh_type, VERTEX, scalar_type> n_type;

	typedef typename std::conditional<EnableImplicit, Field<mesh_type, VERTEX, nTuple<3, scalar_type>>,
	        Field<mesh_type, EDGE, scalar_type> >::type J_type;

	typedef nTuple<8, Real> storage_value_type;

	struct Point_s
	{
		coordinates_type x;
		Vec3 v;
		Real f;
		scalar_type w;

		static std::string DataTypeDesc()
		{
			std::ostringstream os;
			os

			<< "H5T_COMPOUND {          "

			<< "   H5T_ARRAY { [3] H5T_NATIVE_DOUBLE}    \"x\" : " << (offsetof(Point_s, x)) << ";"

			<< "   H5T_ARRAY { [3] H5T_NATIVE_DOUBLE}    \"v\" :  " << (offsetof(Point_s, v)) << ";"

			<< "   H5T_NATIVE_DOUBLE    \"f\" : " << (offsetof(Point_s, f)) << ";"

			<< "   H5T_NATIVE_DOUBLE    \"w\" : " << (offsetof(Point_s, w)) << ";"

			<< "}";

			return os.str();
		}

	};

private:
	Real cmr_, q_kT_;
public:
	mesh_type const &mesh;

public:
	template<typename TDict, typename ...Args>
	PICEngineDeltaF(mesh_type const &pmesh, TDict const& dict, Args const & ...args)
			: mesh(pmesh), m(dict["Mass"].template as<Real>(1.0)), q(dict["Charge"].template as<Real>(1.0)),

			cmr_(q / m), q_kT_(1.0)
	{
		DEFINE_PHYSICAL_CONST(mesh.constants());

		q_kT_ = q / (dict["Temperature"].template as<Real>(1.0) * boltzmann_constant);

	}
	~PICEngineDeltaF()
	{
	}

	static std::string GetTypeAsString()
	{
		return "DeltaF";
	}

	std::string Dump(std::string const & path = "", bool is_verbose = false) const
	{
		std::stringstream os;

		DEFINE_PHYSICAL_CONST(mesh.constants());

		os << "Engine = '" << GetTypeAsString() << "' "

		<< " , " << "Mass = " << m / proton_mass << " * m_p"

		<< " , " << "Charge = " << q / elementary_charge << " * q_e"

		<< " , " << "Temperature = " << q / q_kT_ / elementary_charge << "* eV"

		;

		return os.str();
	}

	static Point_s DefaultValue()
	{
		Point_s p;
		p.f = 1.0;
		p.w = 0.0;
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
	// x(-1/2->1/2), w(-1/2,1/2)
	template<typename TJ, typename TE, typename TB, typename ... Others>
	inline void NextTimeStepZero(Bool2Type<true>, Point_s * p, Real dt, TJ *J, TE const &fE, TB const & fB,
	        Others const &...others) const
	{
		p->x += p->v * dt * 0.5;

//		auto B = interpolator_type::Gather(fB, p->x);
		auto E = interpolator_type::Gather(fE, p->x);

		auto a = (-Dot(E, p->v) * q_kT_ * dt);
		p->w = (-a + (1 + 0.5 * a) * p->w) / (1 - 0.5 * a);

		p->x += p->v * dt * 0.5;

		Vec3 v;
		v = p->v * p->f * p->w * q;
		interpolator_type::Scatter(p->x, v, J);

	}
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
	template<typename TV, typename TE, typename TB, typename ... Others>
	inline void NextTimeStepZero(Bool2Type<false>, Point_s * p, Real dt, Field<mesh_type, EDGE, TV> *J, TE const &fE,
	        TB const & fB, Others const &...others) const
	{
		p->x += p->v * dt * 0.5;
		auto B = interpolator_type::Gather(fB, p->x);
		auto E = interpolator_type::Gather(fE, p->x);

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

		Vec3 v;
		v = p->v * p->f * p->w * q;
		interpolator_type::Scatter(p->x, v, J);
	}
	template<typename TE, typename TB, typename ... Others>
	inline void NextTimeStepHalf(Bool2Type<false>, Point_s * p, Real dt, TE const &fE, TB const & fB,
	        Others const &...others) const
	{

	}
	template<typename TV, typename ...Args>
	void Scatter(Point_s const & p, Field<mesh_type, VERTEX, TV> * n, Args const & ...) const
	{
		interpolator_type::Scatter(p.x, q * p.f * p.w, n);
	}
	inline Real PullBack(Point_s const & p, nTuple<3, Real> *x, nTuple<3, Real> * v) const
	{
		*x = p.x;
		*v = p.v;
		return p.f * p.w;
	}
	inline void PushForward(nTuple<3, Real> const&x, nTuple<3, Real> const& v, Point_s * p) const
	{
		p->x = x;
		p->v = v;
	}
	static inline Point_s make_point(coordinates_type const & x, Vec3 const &v, Real f)
	{
		return std::move(Point_s( { x, v, f, 0 }));
	}

};

template<typename ... TS> std::ostream&
operator<<(std::ostream& os, typename PICEngineDeltaF<TS...>::Point_s const & p)
{
	os << "{ x= {" << p.x << "} , v={" << p.v << "}, f=" << p.f << " , w=" << p.w << " }";

	return os;
}

} // namespace simpla

#endif /* PIC_ENGINE_DELTAF_H_ */
