/*
 * pic_engine_full.h
 *
 *  Created on: 2013年11月6日
 *      Author: salmon
 */

#ifndef PIC_ENGINE_FULL_H_
#define PIC_ENGINE_FULL_H_

#include <string>

#include "../fetl/primitives.h"
#include "../fetl/ntuple.h"

namespace simpla
{

template<typename TM>
struct PICEngineFull
{

public:
	typedef PICEngineFull<TM> this_type;
	typedef TM mesh_type;
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

	PICEngineFull(mesh_type const &pmesh) :
			mesh(pmesh), m_(1.0), q_(1.0), cmr_(1.0)
	{
	}
	~PICEngineFull()
	{
	}

	static std::string TypeName()
	{
		return "Full";
	}
	std::string GetTypeAsString() const
	{
		return "Full";
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

	template<typename TDict>
	inline void Load(TDict const &dict)
	{
		m_ = dict["Mass"].template as<Real>(1.0);
		q_ = dict["Charge"].template as<Real>(1.0);
		cmr_ = q_ / m_;
	}

	std::ostream & Print(std::ostream & os) const
	{

		os << "Engine = '" << GetTypeAsString() << "' "

		<< " , " << "Mass = " << m_

		<< " , " << "Charge = " << q_;

		return os;
	}

	static inline Point_s DefaultValue()
	{
		Point_s p;
		p.f = 1.0;
		return std::move(p);
	}

	template<typename TB, typename TE, typename ... Others>
	inline void NextTimeStep(Point_s * p, Real dt, TE const &fE, TB const & fB, Others const &...others) const
	{
		BorisMethod(dt, cmr_, fE, fB, &(p->x), &(p->v));
	}

	template<typename TV, typename ... Others>
	inline typename std::enable_if<!is_ntuple<TV>::value, void>::type Scatter(Point_s const &p,
			Field<mesh_type, VERTEX, TV>* n, Others const &... others) const
	{
		mesh.Scatter(p.x, p.f, n);
	}

	template<int IFORM, typename TV, typename ...Others>
	inline void Scatter(Point_s const &p, Field<mesh_type, IFORM, TV>* J, Others const &... others) const
	{
		typename Field<mesh_type, IFORM, TV>::field_value_type v;
		v = p.v * p.f;
		mesh.Scatter(p.x, v, J);
	}

	static inline Point_s make_point(coordinates_type const & x, Vec3 const &v, Real f)
	{
		return std::move(Point_s(
		{ x, v, f }));
	}

};

template<typename OS, typename TM> OS&
operator<<(OS& os, typename PICEngineFull<TM>::Point_s const & p)
{
	os << "{ x= {" << p.x << "} , v={" << p.v << "}, f=" << p.f << " }";

	return os;
}

} // namespace simpla

#endif /* PIC_ENGINE_FULL_H_ */
