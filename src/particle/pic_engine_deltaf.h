/*
 * pic_engine_deltaf.h
 *
 *  Created on: 2013年12月10日
 *      Author: salmon
 */

#ifndef PIC_ENGINE_DELTAF_H_
#define PIC_ENGINE_DELTAF_H_

#include <string>

#include "../fetl/primitives.h"
#include "../fetl/ntuple.h"

namespace simpla
{

template<typename TM>
struct PICEngineDeltaF
{

public:
	typedef PICEngineDeltaF<TM> this_type;
	typedef TM mesh_type;
	typedef typename mesh_type::coordinates_type coordinates_type;
	typedef typename mesh_type::scalar_type scalar_type;
	typedef nTuple<7, Real> storage_value_type;

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
public:
	mesh_type const &mesh;

public:
	PICEngineDeltaF(mesh_type const &pmesh) :
			mesh(pmesh), m_(1.0), q_(1.0), cmr_(1.0), q_kT_(1.0)
	{
	}
	~PICEngineDeltaF()
	{
	}

	static std::string TypeName()
	{
		return "DeltaF";
	}
	std::string GetTypeAsString() const
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
	size_t GetAffectedRegion() const
	{
		return 2;
	}
	void Load(LuaObject const &vm)
	{

		DEFINE_PHYSICAL_CONST(mesh.constants());

		m_ = vm["Mass"].as<Real>();
		q_ = vm["Charge"].as<Real>();
		cmr_ = q_ / m_;
		q_kT_ = q_ / (vm["T"].as<Real>() * boltzmann_constant);
	}

	std::ostream & Save(std::ostream & os) const
	{

		DEFINE_PHYSICAL_CONST(mesh.constants());

		os << "Engine = '" << GetTypeAsString() << "' "

		<< " , " << "Mass = " << m_

		<< " , " << "Charge = " << q_

		<< " , " << "T = " << q_ / q_kT_ / elementary_charge << " eV"

		;

		return os;
	}
	void Update()
	{
	}
	static Point_s DefaultValue()
	{
		Point_s p;
		p.f = 1.0;
		p.w = 0.0;
		return std::move(p);
	}

	template<typename TB, typename TE, typename ... Others> inline
	void NextTimeStep(Point_s * p, Real dt, TB const & fB, TE const &fE, Others const &...others) const
	{
		BorisMethod(dt, cmr_, fB, fE, &(p->x), &(p->v));

		// FIXME miss one term E\cross B \cdot \Grad n
		auto a = (-Dot(fE(p->x), p->v) * q_kT_ * dt);

		p->w = (-a + (1 + 0.5 * a) * p->w) / (1 - 0.5 * a);

//		p->w += (1.0 - p->w) * (-q_ * InnerProduct(fE(p->x), p->v) / T_ * dt);
	}

	template<typename TV, typename ... Others>
	inline typename std::enable_if<!is_ntuple<TV>::value, void>::type Scatter(Point_s const &p,
			Field<mesh_type, VERTEX, TV>* n, Others const &... others) const
	{
		mesh.Scatter(p.x, p.f * p.w, n);
	}

	template<int IFORM, typename TV, typename ...Others>
	inline void Scatter(Point_s const &p, Field<mesh_type, IFORM, TV>* J, Others const &... others) const
	{
		mesh.Scatter(p.x, p.v * (p.f * p.w), J);
	}

	static inline Point_s make_point(coordinates_type const & x, Vec3 const &v, Real f)
	{
		return std::move(Point_s(
		{ x, v, f }));
	}

};

template<typename OS, typename TM> OS&
operator<<(OS& os, typename PICEngineDeltaF<TM>::Point_s const & p)
{
	os << "{ x= {" << p.x << "} , v={" << p.v << "}, f=" << p.f << " , w=" << p.w << " }";

	return os;
}

} // namespace simpla

#endif /* PIC_ENGINE_DELTAF_H_ */
