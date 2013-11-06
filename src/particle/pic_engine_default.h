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
template<typename XCoordinates>
struct PICEngineDefault
{
	typedef XCoordinates coordinates_type;

	struct Point_s
	{
		coordinates_type x;
		Vec3 v;
		Real f;
	};

	static void SetDefaultValue(Point_s & p)
	{
		p.f = 1.0;
	}

	template<typename TB, typename TE>
	static inline void Push(Point_s & p, Real m, Real q, TB const & fB,
			TE const &fE)
	{
		auto B = fB(p.x);
		auto E = fE(p.x);
	}

	template<typename TB, typename TE, typename TJ>
	static inline void ScatterJ(Point_s & p, Real m, Real q, TJ & fJ,
			TB const & pB, TE const &pE)
	{
		fJ.Scatter(p.v * p.f, p.x);
	}

	template<typename TN, typename TB, typename TE>
	static inline void ScatterN(Point_s & p, Real m, Real q, TN & fn,
			TB const & pB, TE const &pE)
	{
		fn.Scatter(p.f, p.x);
	}

	template<typename TX, typename TV, typename TN, typename ...Args>
	static inline void CoordTrans(Point_s & p, Real m, Real q, TX const & x,
			TV const &v, TN const & n, Args...)
	{
		p.x = x;
		p.v = v;
		p.f *= n(p.x);
	}
};
}  // namespace simpla

#endif /* PIC_ENGINE_DEFAULT_H_ */
