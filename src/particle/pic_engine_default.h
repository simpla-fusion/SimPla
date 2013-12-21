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

	struct Point_s
	{
		coordinates_type x;
		Vec3 v;
		scalar_type f;
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

//#include <H5Cpp.h>
//template<typename > struct HDF5DataType;
//template<typename TM>
//struct HDF5DataType<typename PICEngineDefault<TM>::Point_s>
//{
////	H5::DataType operator()
////	{
////		char desc[1024];
////		snprintf(desc, sizeof(desc), "H5T_COMPOUND {          "
////				"   H5T_ARRAY { [3] H5T_NATIVE_DOUBLE}    \"X\" : %ul;"
////				"   H5T_ARRAY { [3] H5T_NATIVE_DOUBLE}    \"V\" : %ul;"
////				"   H5T_NATIVE_DOUBLE    \"F\" : %u;"
////				"   H5T_ARRAY { [%d] H5T_NATIVE_DOUBLE}    \"w\" : %d;"
////				"}", (offsetof(Point_s, X)),
////				(offsetof(Point_s, V)),
////				(offsetof(Point_s, F)),
////				num_of_mate,
////				(offsetof(Point_s, w)));
////	}
//};
}// namespace simpla

#endif /* PIC_ENGINE_DEFAULT_H_ */
