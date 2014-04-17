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

template<typename TM, typename Interpolator = typename TM::interpolator_type>
struct PICEngineDefault
{

public:
	typedef PICEngineDefault<TM> this_type;
	typedef TM mesh_type;
	typedef Interpolator interpolator_type;

	typedef typename mesh_type::coordinates_type coordinates_type;
	typedef typename mesh_type::scalar_type scalar_type;

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
	Real m_, q_, cmr_;
public:
	mesh_type const &mesh;

public:

	template<typename ...Args>
	PICEngineDefault(mesh_type const &pmesh, Args const & ...args)
			: mesh(pmesh), m_(1.0), q_(1.0), cmr_(1.0)
	{
		Load(std::forward<Args const &>(args)...);
	}
	PICEngineDefault(mesh_type const &pmesh)
			: mesh(pmesh), m_(1.0), q_(1.0), cmr_(1.0)
	{
	}
	~PICEngineDefault()
	{
	}

	static std::string TypeName()
	{
		return "Default";
	}
	static std::string GetTypeAsString()
	{
		return "Default";
	}
	size_t GetAffectedRange() const
	{
		return 2;
	}

	Real GetMass() const
	{
		return m_;
	}

	Real GetCharge() const
	{
		return q_;
	}

	template<typename TDict, typename ...Others>
	inline void Load(TDict const &dict, Others const &...)
	{
		m_ = dict["Mass"].template as<Real>(1.0);
		q_ = dict["Charge"].template as<Real>(1.0);
		cmr_ = q_ / m_;
	}

	void Print(std::ostream & os) const
	{

		DEFINE_PHYSICAL_CONST(mesh.constants());

		os << "Engine = '" << GetTypeAsString() << "' "

		<< " , " << "Mass = " << m_ / proton_mass << " * m_p"

		<< " , " << "Charge = " << q_ / elementary_charge << " * q_e"

		;

	}
	static inline Point_s DefaultValue()
	{
		Point_s p;
		p.f = 1.0;
		return std::move(p);
	}

	template<typename TJ, typename TB, typename TE, typename ... Others>
	inline void NextTimeStep(Point_s * p, Real dt, TJ *J, TE const &fE, TB const & fB, Others const &...others) const
	{
		BorisMethod(dt, cmr_, interpolator_type::Gather(fE, p->x), interpolator_type::Gather(fB, p->x), &(p->x),
		        &(p->v));

		Scatter(*p, J, fE, fB, std::forward<Others const &>(others)...);
	}
	template<typename TJ, typename ...Args>
	void Scatter(Point_s const & p, Field<mesh_type, EDGE, TJ> * J, Args const & ...) const
	{
		typename Field<mesh_type, EDGE, TJ>::field_value_type v;

		v = p.v * q_ * p.f;

		interpolator_type::Scatter(p.x, v, J);
	}

	template<typename TJ, typename ...Args>
	void Scatter(Point_s const & p, Field<mesh_type, VERTEX, TJ> * n, Args const & ...) const
	{
		interpolator_type::Scatter(p.x, q_ * p.f, n);
	}

	static inline Point_s make_point(coordinates_type const & x, Vec3 const &v, Real f)
	{
		return std::move(Point_s( { x, v, f }));
	}

};

template<typename OS, typename TM> OS&
operator<<(OS& os, typename PICEngineDefault<TM>::Point_s const & p)
{
	os << "{ x= {" << p.x << "} , v={" << p.v << "}, f=" << p.f << " }";

	return os;
}

} // namespace simpla

#endif /* PIC_ENGINE_DEFAULT_H_ */
