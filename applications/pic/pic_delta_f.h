/*
 * pic_delta_f.h
 *
 *  Created on: 2013年10月15日
 *      Author: salmon
 */

#ifndef PIC_DELTA_F_H_
#define PIC_DELTA_F_H_
namespace simpla
{
struct PStr_DeltaF
{
	nTuple<3, Real> x;
	nTuple<3, Real> v;
	Real f;
	Real w;
};

template<typename TM>
struct PICEngineDeltaF
{

private:
	Real m_, q_;

public:
	typedef TM mesh_type;
	typedef typename mesh_type::coordinates_type coordinates_type;
	typedef typename mesh_type::scalar scalar;

public:

	mesh_type const &mesh;

	struct Point_s
	{
		coordinates_type x;
		nTuple<3, Real> v;
		Real f;
		scalar w;
	};

	PICEngineDeltaF(mesh_type const &pmesh) :
			mesh(pmesh), m_(1.0), q_(1.0)
	{

	}
	~PICEngineDeltaF()
	{
	}
	static inline std::string TypeName()
	{
		return "DeltaF";
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
		p.w = 0;
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
		n.Scatter(p.v * p.f * (1 + p.w), p.x);
	}

	template<typename TN, typename ... Args>
	inline void Collect(Int2Type<0>, Point_s const &p, TN & n,
			Args const& ... args) const
	{
		n.Scatter(p.f * (1 + p.w), p.x);
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

#endif /* PIC_DELTA_F_H_ */
