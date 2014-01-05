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
#include "../fetl/ntuple_ops.h"

namespace simpla
{

template<typename > class PICEngineBase;

template<typename TM>
struct PICEngineDeltaF: public PICEngineBase<TM>
{
	Real cmr_, q_, T_;
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
			os

			<< "H5T_COMPOUND {          "

			<< "   H5T_ARRAY { [3] H5T_NATIVE_DOUBLE}    \"x\" : " << (offsetof(Point_s, x)) << ";"

			<< "   H5T_ARRAY { [3] H5T_NATIVE_DOUBLE}    \"v\" :  " << (offsetof(Point_s, v)) << ";"

			<< "   H5T_NATIVE_DOUBLE    \"f\" : " << (offsetof(Point_s, f)) << ";"

			<< "}";

			return os.str();
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
	virtual inline std::string GetTypeAsString() const override
	{
		return "DeltaF";
	}

	inline void Deserialize(LuaObject const &obj) override
	{
		base_type::Deserialize(obj);
		obj["T"].as<Real>(&T_);
		Update();
	}
	void Update() override
	{
		cmr_ = base_type::q_ / base_type::m_;
		q_ = base_type::q_;
	}
	std::ostream & Serialize(std::ostream & os) const
	{

		os << "Engine = '" << GetTypeAsString() << "' " << " , ";

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

	template<typename TB, typename TE, typename ... Others>
	inline void NextTimeStep(Point_s * p, Real dt, TB const & fB, TE const &fE, Others const &...others) const
	{

		BorisMethod(dt, cmr_, fB, fE, &(p->x), &(p->v));

		p->w = (1.0 - p->w) * InnerProduct(fE(p->x), p->v) / T_ * dt;
	}

	template<typename TV, typename ... Others>
	inline typename std::enable_if<!is_ntuple<TV>::value, void>::type Collect(Point_s const &p,
	        Field<Geometry<mesh_type, 0>, TV>* n, Others const &... others) const
	{
		n->Collect(p.f * p.w, p.x);
	}

	template<int IFORM, typename TV, typename ...Others>
	inline void Collect(Point_s const &p, Field<Geometry<mesh_type, IFORM>, TV>* J, Others const &... others) const
	{
		J->Collect(p.v * p.f * p.w, p.x);
	}

	template<typename TX, typename TV, typename TFun>
	inline Point_s Trans(TX const & x, TV const &v, TFun const & n, ...) const
	{
		Point_s p;
		p.x = x;
		p.v = v;
		p.f = n(x);
		p.w = 0.0;

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
operator<<(std::ostream& os, typename PICEngineDeltaF<TM>::Point_s const & p)
{
	os << "{ x= {" << p.x << "} , v={" << p.v << "}, f=" << p.f << " }";

	return os;
}

} // namespace simpla

#endif /* PIC_ENGINE_DELTAF_H_ */
