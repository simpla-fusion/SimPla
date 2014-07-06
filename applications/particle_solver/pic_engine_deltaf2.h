/*
 * pic_engine_deltaf.h
 *
 * \date  2013-12-10
 *      \author  salmon
 */

#ifndef PIC_ENGINE_DELTAF_H_
#define PIC_ENGINE_DELTAF_H_

#include <string>

#include "../utilities/primitives.h"
#include "../utilities/ntuple.h"

namespace simpla
{

template<typename TM, typename TS = Real>
struct PICEngineDeltaF
{

public:
	typedef PICEngineDeltaF<TM, TS> this_type;

	typedef TM mesh_type;
	typedef TS scalar_type;

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
	Real m, cmr_, q, q_k_;
public:
	mesh_type const &mesh;
	Field<mesh_type, VERTEX, Real> n0;
	Field<mesh_type, VERTEX, Real> T0;
	Field<mesh_type, EDGE, Real> gradn0;
	Field<mesh_type, EDGE, Real> gradT0;
public:
	PICEngineDeltaF(mesh_type const &pmesh) :
			mesh(pmesh), m(1.0), q(1.0), cmr_(1.0), q_k_(1.0), n0(mesh), T0(mesh), gradn0(mesh), gradT0(mesh)
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
		return m;
	}

	Real GetCharge() const
	{
		return q;
	}
	size_t GetAffectedRange() const
	{
		return 2;
	}
	template<typename TDict, typename TN, typename TT>
	void Load(TDict const &dict, TN const & n, TT const &T)
	{

		m = dict["Mass"].template as<Real>(1.0);
		q = dict["Charge"].template as<Real>(1.0);
		cmr_ = q / m;
	}

	std::ostream & Print(std::ostream & os) const
	{

		DEFINE_PHYSICAL_CONST;

		os << "Engine = '" << GetTypeAsString() << "' "

		<< " , " << "Mass = " << m

		<< " , " << "Charge = " << q

		<< " , " << "Temperature = " << q / q_k_ / elementary_charge << "* eV"

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
	void NextTimeStep(Point_s * p, Real dt, TE const &fE, TB const & fB, Others const &...others) const
	{
		BorisMethod(dt, cmr_, fE, fB, &(p->x), &(p->v));
		DEFINE_PHYSICAL_CONST;
		// FIXME miss one term E\cross B \cdot \Grad n

		auto T_ = T(p->x);

		nTuple<3, scalar_type> kapp;

		kapp = gradn0(p->x) / n0(p->x)

//		+ (0.5 * gradT0(p->x) / T0(p->x) * m_ * Dot(p->v, p->v)) / (T0(p->x) * boltzmann_constant)

		        - fE(p->x) * q / (T0(p->x) * boltzmann_constant);

		auto a = Dot(kapp, p->v) * dt;

		p->w = 0.5 * (-a + (1 + 0.5 * a) * p->w) / (1 - 0.5 * a);

	}

	template<typename TV, typename ... Others>
	inline typename std::enable_if<!is_ntuple<TV>::value, void>::type Scatter(Point_s const &p,
	        Field<mesh_type, VERTEX, TV>* n, Others const &... others) const
	{
		mesh.Scatter(p.x, p.f * p.w * q, n);
	}

	template<unsigned int IFORM, typename TV, typename ...Others>
	inline void Scatter(Point_s const &p, Field<mesh_type, IFORM, TV>* J, Others const &... others) const
	{
		typename Field<mesh_type, IFORM, TV>::field_value_type v;

		v = p.v * (p.f * p.w * q);
		mesh.Scatter(p.x, v, J);
	}

	static inline Point_s make_point(coordinates_type const & x, Vec3 const &v, Real f)
	{
		return std::move(Point_s( { x, v, f, 0 }));
	}

};

template<typename OS, typename TM, typename TS> OS&
operator<<(OS& os, typename PICEngineDeltaF<TM, TS>::Point_s const & p)
{
	os << "{ x= {" << p.x << "} , v={" << p.v << "}, f=" << p.f << " , w=" << p.w << " }";

	return os;
}

} // namespace simpla

#endif /* PIC_ENGINE_DELTAF_H_ */
