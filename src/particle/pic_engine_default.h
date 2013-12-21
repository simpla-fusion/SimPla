/*
 * pic_engine_default.h
 *
 *  Created on: 2013年11月6日
 *      Author: salmon
 */

#ifndef PIC_ENGINE_DEFAULT_H_
#define PIC_ENGINE_DEFAULT_H_

#include <string>

#include "../fetl/primitives.h"

namespace simpla
{

template<typename > class PICEngineBase;

template<typename TM>
struct PICEngineDefault: public PICEngineBase<TM>
{
public:
	typedef PICEngineBase<TM> base_type;
	typedef PICEngineDefault<TM> this_type;
	typedef TM mesh_type;
	typedef typename mesh_type::coordinates_type coordinates_type;
	typedef typename mesh_type::scalar_type scalar_type;

	typedef nTuple<7, Real> storage_value_type;

	struct Point_s
	{
		coordinates_type x;
		Vec3 v;
		scalar_type f;

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

	PICEngineDefault(mesh_type const &pmesh)
			: base_type(pmesh)
	{

	}
	~PICEngineDefault()
	{
	}

	inline std::string TypeName() const
	{
		return "Default";
	}

	inline void Deserialize(LuaObject const &obj)
	{
		base_type::Deserialize(obj);
	}

	std::ostream & Serialize(std::ostream & os) const
	{

		os << "Engine = 'Default' ,";

		base_type::Serialize(os);

		return os;
	}

	static inline Point_s DefaultValue()
	{
		Point_s p;
		p.f = 1.0;
		return std::move(p);
	}

	template<typename TB, typename TE>
	inline void Push(Point_s & p, Real dt, TB const & fB, TE const &fE) const
	{
		auto B = fB(p.x);
		auto E = fE(p.x);
	}

	template<typename TN, typename ... Args>
	inline void Collect(Int2Type<0>, Point_s const &p, TN * n, Args const& ... args) const
	{
		n->Scatter(p.f, p.x);
	}

	template<typename TJ, typename ... Args>
	inline void Collect(Int2Type<1>, Point_s const &p, TJ * J, Args const& ... args) const
	{
		J->Scatter(p.v * p.f, p.x);
	}

	template<typename TN, typename ... Args>
	inline void Collect(Int2Type<2>, Point_s const &p, TN * n, Args const& ... args) const
	{
	}

	template<typename TX, typename TV, typename TN, typename ...Args>
	inline void CoordTrans(Point_s & p, TX const & x, TV const &v, TN const & n, Args...) const
	{
		p.x = x;
		p.v = v;
		p.f *= n(p.x);
	}

};

template<typename TM> std::ostream&
operator<<(std::ostream& os, typename PICEngineDefault<TM>::Point_s const & p)
{
	os << "{ x= {" << p.x << "} , v={" << p.v << "}, f=" << p.f << " }";

	return os;
}

} // namespace simpla

#endif /* PIC_ENGINE_DEFAULT_H_ */
