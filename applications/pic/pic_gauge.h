/*
 * pic_gague.h
 *
 *  Created on: 2013年10月15日
 *      Author: salmon
 */

#ifndef PIC_GAGUE_H_
#define PIC_GAGUE_H_

#include <fetl/primitives.h>

namespace simpla
{

template<typename TM, int N>
struct PICEngineGGauge
{

private:
	Real m_, q_;

public:
	typedef TM mesh_type;
	typedef typename mesh_type::coordinates_type coordinates_type;
	typedef typename mesh_type::scalar scalar;

public:
	enum
	{
		NUM_OF_MATE = N
	};

	mesh_type const &mesh;

	struct Point_s
	{
		coordinates_type x;
		nTuple<3, Real> v;
		Real f;
		scalar w[];
	};

	PICEngineGGauge(mesh_type const &pmesh) :
			mesh(pmesh), m_(1.0), q_(1.0)
	{

	}
	~PICEngineGGauge()
	{
	}

	inline std::string TypeName()
	{
		return "GGauge" + ToString(NUM_OF_MATE);
	}

	template<typename PT>
	inline void Deserialize(PT const &vm)
	{
		vm.template GetValue<Real>("Mass", &m_);
		vm.template GetValue<Real>("Charge", &q_);
	}

	template<typename PT>
	inline void Serialize(PT &vm) const
	{
		vm.template SetValue<Real>("Mass", m_);
		vm.template SetValue<Real>("Charge", q_);
	}

	static inline Point_s DefaultValue()
	{
		Point_s p;
		p.f = 1.0;
		std::fill(p.w, p.w + NUM_OF_MATE, 0);
		return std::move(p);
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
		n.Scatter(p.f, p.x);
	}
	template<typename TN, typename ... Args>
	inline void Collect(Int2Type<2>, Point_s const &p, TN & n,
			Args const& ... args) const
	{
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

}  // namespace simpla

#endif /* PIC_GAGUE_H_ */
