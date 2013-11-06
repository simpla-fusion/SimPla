/*
 * ggauge.h
 *
 *  Created on: 2013年10月23日
 *      Author: salmon
 */

#ifndef GGAUGE_H_
#define GGAUGE_H_

#include <fetl/primitives.h>
#include <numeric/multi_normal_distribution.h>
#include <numeric/normal_distribution_icdf.h>
#include <numeric/rectangle_distribution.h>

namespace simpla
{

template<typename TX, int NMATE = 8>
struct GGauge
{
	typedef TX coordinates_type;

	struct Point_s
	{
		coordinates_type x;
		Vec3 v;
		Real f;
		Real w[NMATE];

	};

	static void SetDefaultValue(Point_s & p)
	{
		p.f = 1.0;
		std::fill(p.w, p.w + NMATE, 0);
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
			TB const & pB, TE const &pE) const
	{
		fJ.Scatter(p.x, p.v * p.f);
	}

	template<typename TN, typename TB, typename TE>
	static inline void ScatterJ(Point_s & p, Real m, Real q, TN & fn,
			TB const & pB, TE const &pE) const
	{
		fn.Scatter(p.x, p.f);
	}

	template<typename TX, typename TV, typename TN, typename ...Args>
	static inline void CoordTrans(Point_s & p, Real m, Real q, TX const & x,
			TV const &v, TN const & n, Args...) const
	{
		p.x = x;
		p.v = v;
		p.f *= n(p.x);
	}

//	struct Generator
//	{
//		typedef rectangle_distribution<TM::NUM_OF_DIMS> x_dist_type;
//
//		typedef multi_normal_distribution<3, Real, normal_distribution_icdf> v_dist_type;
//
//		x_dist_type x_dist_;
//
//		v_dist_type v_dist_;
//
//		Generator(x_dist_type const & x_dist, v_dist_type const & v_dist) :
//				x_dist_(x_dist), v_dist_(v_dist)
//		{
//		}
//
//		template<typename TTemp, typename TDensity>
//		Generator(
//				std::vector<typename mesh_type::coordinate_type> const& cell_shape,
//				TDensity const & n, TTemp const temp) :
//				x_dist_(cell_shape), v_dist_(temp)
//		{
//		}
//
//		template<typename Generator>
//		inline void operator()(Point_s & p, Generator g) const
//		{
//			p.x = x_dist_(g);
//			p.v = v_dist_(g);
//
//			p.f *= n_(p.x);
//		}
//
//	};

};

} // namespace simpla

#endif /* GGAUGE_H_ */
