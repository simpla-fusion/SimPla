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
	bool isXVSync_;
public:
	mesh_type const &mesh;

public:
	template<typename ...Args>
	PICEngineDeltaF(mesh_type const &pmesh, Args const & ...args)
			: mesh(pmesh), m_(1.0), q_(1.0), cmr_(1.0), q_kT_(1.0), isXVSync_(true)
	{
		Load(std::forward<Args const &>(args)...);
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
	template<typename TDict, typename ...Others>
	void Load(TDict const &dict, Others const &...)
	{

		DEFINE_PHYSICAL_CONST(mesh.constants());

		m_ = dict["Mass"].template as<Real>(1.0);
		q_ = dict["Charge"].template as<Real>(1.0);
		cmr_ = q_ / m_;
		q_kT_ = q_ / (dict["Temperature"].template as<Real>(1.0) * boltzmann_constant);
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

	template<typename TJ, typename TB, typename TE, typename ... Others>
	inline void NextTimeStep(Point_s * p, Real dt, TJ *J, TE const &fE, TB const & fB, Others const &...others) const
	{

		auto B = interpolator_type::Gather(fB, p->x);
		auto E = interpolator_type::Gather(fE, p->x);

		Vec3 v_;

		auto t = B * (cmr_ * dt * 0.5);

		p->v += E * (cmr_ * dt * 0.5);

		v_ = p->v + Cross(p->v, t);

		v_ = Cross(v_, t) / (Dot(t, t) + 1.0);

		p->v += v_;

		// FIXME miss one term E\cross B \cdot \Grad n
		// @NOTE Nonlinear delta-f
		auto a = (-Dot(E, p->v) * q_kT_ * dt);
		p->w = (-a + (1 + 0.5 * a) * p->w) / (1 - 0.5 * a);

		p->v += v_;

		p->v += E * (cmr_ * dt * 0.5);

		p->x += p->v * dt * 0.5;
		Vec3 v;
		v = p->v * p->f * p->w;
		interpolator_type::Scatter(p->x, v, J);
		p->x += p->v * dt * 0.5;
	}

	template<typename TJ, typename ...Args>
	void Scatter(Point_s const & p, Field<mesh_type, EDGE, TJ> * J, Args const & ...) const
	{
		typename Field<mesh_type, EDGE, TJ>::field_value_type v;

		v = p.v * q_ * p.f * p.w;

		interpolator_type::Scatter(p.x, v, J);
	}

	template<typename TJ, typename ...Args>
	void Scatter(Point_s const & p, Field<mesh_type, VERTEX, TJ> * n, Args const & ...) const
	{
		interpolator_type::Scatter(p.x, q_ * p.f * p.w, n);
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
