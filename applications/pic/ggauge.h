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

template<typename TM, int NMATE = 8>
struct GGauge
{
	typedef TM mesh_type;
	typedef typename mesh_type::index_type index_type;

	struct Point_s
	{
		typename TM::coordinates_type x;
		Vec3 v;
		Real f;
		Real w[NMATE];

		Point_s() :
				f(1.0)
		{
			std::fill(w, w + NMATE, 0);
		}
	}
	struct Push
	{
		template<typename TB, typename TE>
		inline void operator()(Point_s & p, TB const & fB, TE const &fE)
		{
			auto B = fB(p.x);
			auto E = fE(p.x);
		}
	}
	struct ScatterJ
	{
		template<typename TB, typename TE, typename TJ>
		inline void operator()(Point_s & p, TB const & pB, TE const &pE,
				TJ & fJ) const
		{
			fJ.Scatter(p.x, p.v);
		}
	};

	struct Generator
	{
		typedef rectangle_distribution<TM::NUM_OF_DIMS> x_dist_type;

		typedef multi_normal_distribution<3, Real, normal_distribution_icdf> v_dist_type;

		x_dist_type x_dist_;

		v_dist_type v_dist_;

		Generator(x_dist_type const & x_dist, v_dist_type const & v_dist) :
				x_dist_(x_dist), v_dist_(v_dist)
		{
		}

		template<typename TTemp, typename TDensity>
		Generator(
				std::vector<typename mesh_type::coordinate_type> const& cell_shape,
				TDensity const & n, TTemp const temp) :
				x_dist_(cell_shape), v_dist_(temp)
		{
		}

		template<typename Generator>
		inline void operator()(Point_s & p, Generator g) const
		{
			p.x = x_dist_(g);
			p.v = v_dist_(g);

			p.f *= n_(p.x);
		}

	};

};

} // namespace simpla

#endif /* GGAUGE_H_ */
