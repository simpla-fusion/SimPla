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

template<typename TM, int DeltaF = 0>
struct PICEngineFull
{

public:
	typedef PICEngineBase<TM> base_type;
	typedef PICEngineFull<TM, DeltaF> this_type;
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

	PICEngineFull(mesh_type const &pmesh)
			: mesh(pmesh), m_(1.0), q_(1.0), cmr_(1.0)
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
	size_t GetAffectedRegion() const
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

	inline void Load(LuaObject const &vm)
	{
		m_ = vm["Mass"].as<Real>();
		q_ = vm["Charge"].as<Real>();
		cmr_ = q_ / m_;
	}

	std::ostream & Save(std::ostream & os) const
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
	inline void NextTimeStep(Point_s * p, Real dt, TB const & fB, TE const &fE, Others const &...others) const
	{
		BorisMethod(dt, cmr_, fB, fE, &(p->x), &(p->v));
	}

	template<typename TV, typename ... Others>
	inline typename std::enable_if<!is_ntuple<TV>::value, void>::type Collect(Point_s const &p,
	        Field<mesh_type, VERTEX, TV>* n, Others const &... others) const
	{
		n->Collect(p.f, p.x);
	}

	template<int IFORM, typename TV, typename ...Others>
	inline void Collect(Point_s const &p, Field<mesh_type, IFORM, TV>* J, Others const &... others) const
	{
		J->Collect(p.v * p.f, p.x);
	}

	template<typename TX, typename TV, typename TFun>
	inline Point_s Trans(TX const & x, TV const &v, TFun const & n, ...) const
	{
		Point_s p;
		p.x = x;
		p.v = v;
		p.f = n(x);

		return std::move(p);
	}

	template<typename TX, typename TV, typename ... Others>
	inline void Trans(TX const & x, TV const &v, Point_s * p, Others const &...) const
	{
		p->x = x;
		p->v = v;
	}

	template<typename TX, typename TV, typename ... Others>
	inline void InvertTrans(Point_s const &p, TX * x, TV *v, Others const &...) const
	{
		*x = p.x;
		*v = p.v;
	}
};

template<typename TM> std::ostream&
operator<<(std::ostream& os, typename PICEngineFull<TM>::Point_s const & p)
{
	os << "{ x= {" << p.x << "} , v={" << p.v << "}, f=" << p.f << " }";

	return os;
}

} // namespace simpla

#endif /* PIC_ENGINE_FULL_H_ */
