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
namespace simpla
{

template<typename > class PICEngineBase;

template<typename TM>
struct PICEngineDeltaF: public PICEngineBase<TM>
{
	Real cmr_, q_;
	Real T_;
public:
	typedef PICEngineBase<TM> base_type;
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

			//TODO: add complex support
			os

			<< "H5T_COMPOUND {          "

			<< "   H5T_ARRAY { [3] H5T_NATIVE_DOUBLE}    \"x\" : " << (offsetof(Point_s, x)) << ";"

			<< "   H5T_ARRAY { [3] H5T_NATIVE_DOUBLE}    \"v\" :  " << (offsetof(Point_s, v)) << ";"

			<< "   H5T_NATIVE_DOUBLE    \"f\" : " << (offsetof(Point_s, f)) << ";"

			<< "   H5T_NATIVE_DOUBLE    \"w\" : " << (offsetof(Point_s, w)) << ";"

			<< "}";

			return os.str();
		}

		template<typename TX, typename TV, typename TF> inline
		static void Trans(Point_s *p, TX const &x, TV const &v, TF f)
		{
			p->x = x;
			p->v = v;
			p->f = f;
			p->w = 0.0;
		}
	};

	PICEngineDeltaF(mesh_type const &pmesh)
			: base_type(pmesh), cmr_(1.0), q_(1.0), T_(1.0)
	{

	}
	virtual ~PICEngineDeltaF()
	{
	}

	static inline std::string TypeName()
	{
		return "DeltaF";
	}

	virtual inline std::string _TypeName() const
	{
		return this_type::TypeName();
	}

	inline void Deserialize(LuaObject const &obj)
	{
		if (obj.empty())
			return;

		base_type::Deserialize(obj);

		T_ = obj["T"].as<Real>();

	}

	void Update()
	{
		cmr_ = base_type::q_ / base_type::m_;
		q_ = base_type::q_;
	}

	std::ostream & Serialize(std::ostream & os) const
	{

		os << "Engine =" << TypeName() << " , T = " << T_ << " , ";

		base_type::Serialize(os);

		return os;
	}

	static inline Point_s DefaultValue()
	{
		Point_s p;
		p.f = 1.0;
		p.w = 0.0;
		return std::move(p);
	}

	template<typename TB, typename TE, typename ...Others>
	inline void NextTimeStep(Point_s * p, Real dt, TB const & fB, TE const &fE, Others const &...others) const
	{
		// keep x,v at same time step
		p->x += p->v * 0.5 * dt;

		auto B = real(fB(p->x));
		auto E = fE(p->x);

		auto rE = real(E);

		///  @ref  Birdsall(1991)   p.62

		Vec3 v_;

		auto t = B * (cmr_ * dt * 0.5);

		p->v += rE * (cmr_ * dt * 0.5);

		v_ = p->v + Cross(p->v, t);

		p->v += Cross(v_, t) * (2.0 / (Dot(t, t) + 1.0));

		p->v += rE * (cmr_ * dt * 0.5);

		p->x += p->v * 0.5 * dt;

		p->w = (-p->w + 1.0) * Dot(E, p->v) / T_ * dt;

	}

	template<typename ...Others>
	inline void Collect(Point_s const &p, Field<Geometry<mesh_type, 0>, scalar_type>* n, Others const &...others) const
	{
//		n->Collect(p.f * p.w, p.x);
	}

	template<int IFORM, typename TV, typename ... Others>
	inline void Collect(Point_s const &p, Field<Geometry<mesh_type, IFORM>, TV>* J, Others const &...others) const
	{
//		J->Collect(p.v * (p.f * p.w), p.x);
	}

	template<typename TX, typename TV>
	inline Point_s Trans(TX const & x, TV const &v, Real f, ...) const
	{
		Point_s p;
		p.x = x;
		p.v = v;
		p.f = f;
		p.w = 0;
		return std::move(p);
	}

};

template<typename TM> std::ostream&
operator<<(std::ostream& os, typename PICEngineDeltaF<TM>::Point_s const & p)
{
	os << "{ x= {" << p.x << "} , v={" << p.v << "}, f=" << p.f << ", w=" << p.w << " }";

	return os;
}

} // namespace simpla

#endif /* PIC_ENGINE_DELTAF_H_ */
