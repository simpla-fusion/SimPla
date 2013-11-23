/*
 * pic_engine_default.h
 *
 *  Created on: 2013年11月6日
 *      Author: salmon
 */

#ifndef PIC_ENGINE_DEFAULT_H_
#define PIC_ENGINE_DEFAULT_H_

#include <fetl/primitives.h>

namespace simpla
{

template<typename TM>
struct PICEngineDefault
{
public:
	typedef TM mesh_type;
	typedef typename mesh_type::coordinates_type coordinates_type;

	mesh_type const &mesh;

private:
	Real m_, q_;

public:
	struct Point_s
	{
		coordinates_type x;
		Vec3 v;
		Real f;
	};

	template<typename TP>
	PICEngineDefault(mesh_type const &pmesh, TP const & pt) :
			mesh(pmesh)
	{

	}
	~PICEngineDefault()
	{
	}

	template<typename TP>
	inline void SetProperties(TP const &p)
	{
		m_ = p.template Get<Real>("Mass");
		q_ = p.template Get<Real>("Charge");
	}

	template<typename TP>
	inline void GetProperties(TP &p) const
	{
		p.Set("Mass", m_);
		p.Set("Charge", q_);
	}

	template<typename TP>
	inline TP GetProperties() const
	{
		TP p;
		p.Set("Mass", m_);
		p.Set("Charge", q_);
		return std::move(p);
	}

	static void SetDefaultValue(Point_s & p)
	{
		p.f = 1.0;
	}

	template<typename TB, typename TE>
	inline void Push(Point_s & p, TB const & fB, TE const &fE) const
	{
		auto B = fB(p.x);
		auto E = fE(p.x);
	}

	template<typename TN, typename ... Args>
	inline void Collect(Int2Type<1>, Point_s const &p, TN & n,
			Args const& ... args) const
	{
		n.Scatter(p.v * p.f, p.x);
	}

	template<typename TN, typename ... Args>
	inline void Collect(Int2Type<0>, Point_s const &p, TN & n,
			Args const& ... args) const
	{
		n.Scatter(p.v * p.f, p.x);
	}

	template<typename TX, typename TV, typename TN, typename ...Args>
	inline void CoordTrans(Point_s & p, TX const & x, TV const &v, TN const & n,
			Args...) const
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
