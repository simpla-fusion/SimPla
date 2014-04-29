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

template<typename TM, typename TS = Real, typename Interpolator = typename TM::interpolator_type>
struct PICEngineDeltaF
{

public:
	typedef PICEngineDeltaF<TM, TS, Interpolator> this_type;

	typedef TM mesh_type;
	typedef TS scalar_type;
	typedef Interpolator interpolator_type;

	typedef typename mesh_type::coordinates_type coordinates_type;

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
	Real m_, cmr_, q_, q_kT_;
	bool enableImplicit_;
public:
	mesh_type const &mesh;

public:
	template<typename TDict, typename ...Args>
	PICEngineDeltaF(mesh_type const &pmesh, TDict const& dict, Args const & ...args)
			: mesh(pmesh), m_(1.0), q_(1.0), cmr_(1.0), q_kT_(1.0), enableImplicit_(false)
	{
		DEFINE_PHYSICAL_CONST(mesh.constants());

		m_ = dict["Mass"].template as<Real>(1.0);
		q_ = dict["Charge"].template as<Real>(1.0);
		cmr_ = q_ / m_;
		q_kT_ = q_ / (dict["Temperature"].template as<Real>(1.0) * boltzmann_constant);
		enableImplicit_ = dict["EnableImplicit"].template as<bool>(false);

	}
	~PICEngineDeltaF()
	{
	}

	static std::string TypeName()
	{
		return "DeltaF";
	}
	static std::string GetTypeAsString()
	{
		return "DeltaF";
	}

	Real GetMass() const
	{
		return m_;
	}

	Real GetCharge() const
	{
		return q_;
	}
	size_t GetAffectedRange() const
	{
		return 2;
	}

	bool EnableImplicit() const
	{
		return enableImplicit_;
	}
	std::string Dump(std::string const & path = "", bool is_verbose = false) const
	{
		std::stringstream os;

		DEFINE_PHYSICAL_CONST(mesh.constants());

		os << "Engine = '" << GetTypeAsString() << "' "

		<< " , " << "Mass = " << m_ / proton_mass << " * m_p"

		<< " , " << "Charge = " << q_ / elementary_charge << " * q_e"

		<< " , " << "Temperature = " << q_ / q_kT_ / elementary_charge << "* eV"

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

	// x(-1/2->1/2), w(-1/2,1/2)
	template<typename TE, typename TB, typename ... Others>
	inline void NextTimeStepZero(Point_s * p, Real dt, Field<mesh_type, VERTEX, nTuple<3, scalar_type> > *J,
	        TE const &fE, TB const & fB, Others const &...others) const
	{
		p->x += p->v * dt;

//		auto B = interpolator_type::Gather(fB, p->x);
		auto E = interpolator_type::Gather(fE, p->x);

		auto a = (-Dot(E, p->v) * q_kT_ * dt);
		p->w = (-a + (1 + 0.5 * a) * p->w) / (1 - 0.5 * a);

		Vec3 v;
		v = p->v * p->f * p->w;
		interpolator_type::Scatter(p->x, v, J);

		p->x += p->v * dt * 0.5;

	}

	template<typename TV, typename TE, typename TB, typename ... Others>
	inline void NextTimeStepZero(Point_s * p, Real dt, Field<mesh_type, EDGE, TV> *J, TE const &fE, TB const & fB,
	        Others const &...others) const
	{
		p->x += p->v * dt * 0.5;

//		auto B = interpolator_type::Gather(fB, p->x);
		auto E = interpolator_type::Gather(fE, p->x);

		auto a = (-Dot(E, p->v) * q_kT_ * dt);
		p->w = (-a + (1 + 0.5 * a) * p->w) / (1 - 0.5 * a);

		p->x += p->v * dt * 0.5;

		Vec3 v;
		v = p->v * p->f * p->w;
		interpolator_type::Scatter(p->x, v, J);

	}

	template<typename TE, typename TB, typename ... Others>
	inline void NextTimeStepHalf(Point_s * p, Real dt, TE const &fE, TB const & fB, Others const &...others) const
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

	template<typename TV, typename ...Args>
	void Scatter(Point_s const & p, Field<mesh_type, VERTEX, TV> * n, Args const & ...) const
	{
		interpolator_type::Scatter(p.x, q_ * p.f * p.w, n);
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

template<typename TM, typename TS> std::ostream&
operator<<(std::ostream& os, typename PICEngineDeltaF<TM, TS>::Point_s const & p)
{
	os << "{ x= {" << p.x << "} , v={" << p.v << "}, f=" << p.f << " , w=" << p.w << " }";

	return os;
}

} // namespace simpla

#endif /* PIC_ENGINE_DELTAF_H_ */
