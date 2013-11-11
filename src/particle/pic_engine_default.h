/*
 * pic_engine_default.h
 *
 *  Created on: 2013年11月6日
 *      Author: salmon
 */

#ifndef PIC_ENGINE_DEFAULT_H_
#define PIC_ENGINE_DEFAULT_H_

namespace simpla
{

template<typename TM>
struct PICEngineDefault
{
public:
	typedef TM mesh_type;
	typedef typename mesh_type::coordinates_type coordinates_type;

	mesh_type const &mesh;

	Real m_, q_;

	struct Point_s
	{
		coordinates_type x;
		Vec3 v;
		Real f;
	};

	PICEngineDefault(mesh_type const &pmesh) :
			mesh(pmesh), m_(1.0), q_(1.0)
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

	static void SetDefaultValue(Point_s & p)
	{
		p.f = 1.0;
	}

	template<typename TB, typename TE>
	inline void Push(Point_s & p, TB const & fB, TE const &fE)const
	{
		auto B = fB(p.x);
		auto E = fE(p.x);

	}

	template<typename TJ, typename TB, typename TE>
	inline void ScatterJ(Point_s const& p, TJ & fJ, TB const & pB, TE const &pE)const
	{
		fJ.Scatter(p.v * p.f, p.x);
	}

	template<typename TN, typename TB, typename TE>
	inline void ScatterN(Point_s const& p, TN & fn, TB const & pB, TE const &pE)const
	{
		fn.Scatter(p.f, p.x);
	}

	template<typename TX, typename TV, typename TN, typename ...Args>
	inline void CoordTrans(Point_s & p, TX const & x, TV const &v, TN const & n,
			Args...)const
	{
		p.x = x;
		p.v = v;
		p.f *= n(p.x);
	}
};
}  // namespace simpla

#endif /* PIC_ENGINE_DEFAULT_H_ */
