/*
 * pic_engine_deltaf.h
 *
 *  Created on: 2013年12月10日
 *      Author: salmon
 */

#ifndef PIC_ENGINE_DELTAF_H_
#define PIC_ENGINE_DELTAF_H_

namespace simpla
{

template<typename TM>
struct PICEngineDeltaF
{

private:
	Real m_, q_;

public:
	typedef TM mesh_type;
	typedef typename mesh_type::coordinates_type coordinates_type;
	typedef typename mesh_type::scalar scalar;

	mesh_type const &mesh;

	struct Point_s
	{
		coordinates_type x;
		Vec3 v;
		scalar f;
	};

	PICEngineDeltaF(mesh_type const &pmesh)
			: mesh(pmesh), m_(1.0), q_(1.0)
	{

	}

	~PICEngineDeltaF()
	{
	}

	inline Real GetMass() const
	{
		return m_;
	}

	inline Real GetCharge() const
	{
		return q_;
	}
	static inline std::string TypeName()
	{
		return "Default";
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

	std::ostream & Serialize(std::ostream & os) const
	{
		os << "{"

		<< "Engine = 'DeltaF' ,"

		<< "m = " << m_ << " , "

		<< "q = " << q_ << ","

		<< "}";

		return os;
	}
	static inline Point_s DefaultValue()
	{
		Point_s p;
		p.f = 1.0;
		return std::move(p);
	}

	template<typename TB, typename TE>
	inline void Push(Point_s & p, TB const & fB, TE const &fE) const
	{
		auto B = fB(p.x);
		auto E = fE(p.x);
	}

	template<typename TJ, typename ... Args>
	inline void Collect(Int2Type<0>, Point_s const &p, TJ * n, Args const& ... args) const
	{
		n->Scatter(p.f, p.x);
	}

	template<typename TJ, typename ... Args>
	inline void Collect(Int2Type<1>, Point_s const &p, TJ * n, Args const& ... args) const
	{
		n->Scatter(p.v * p.f, p.x);
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
} //namespace simpla
#endif /* PIC_ENGINE_DELTAF_H_ */
