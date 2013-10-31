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
#include <particle/particle.h>

namespace simpla
{

template<typename TM, int NMATE = 8>
struct GGauge_s
{
	typename TM::coordinates_type x;
	Vec3 v;
	Real f;
	Real w[NMATE];

	GGauge_s() :
			f(1.0)
	{
		std::fill(w, w + NMATE, 0);
	}

};

template<typename TM, int NMATE>
class Particle<TM, GGauge_s<TM, NMATE> > : public ParticleBase<TM,
		GGauge_s<TM, NMATE> >
{
public:

	template<typename TM, int NMATE, typename Generator, typename TF>
	void InitLoad(int pic, Real T, TF const & pn, Generator g = Generator())
	{

		rectangle_distribution<TM::NUM_OF_DIMS> x_dist;

		multi_normal_distribution<3, Real, normal_distribution_icdf> v_dist(T);

		this->ForEachCell(

		[&]( cell_type & cell, index_type const &s)
		{
			Real coeff = static_cast<Real>(pic) * mesh_GetDCellVolume(s);

			cell.resize(pic, value_type ());

			auto n = MakeCache(pn, s);

			for (auto & p : cell)
			{
				p.x = mesh_.GetGlobalCoordinates(s, x_dist(g));
				p.v = v_dist(g);
				p.f = coeff * n(p.x);
			}

		}

		);
	}

	template<typename TB, typename TE>
	inline void Push(TB const & pB, TE const &pE)
	{
		ForEachCell(

		[&pB,&pE](cell_type & cell,index_type const &hint_idx)
		{
			auto fB = MakeCache(pB, hint_idx);
			auto fE = MakeCache(pE, hint_idx);

			for (auto const & p : cell)
			{
				auto B = fB(p.x);
				auto E = fE(p.x);
			}
		});

		Sort();
	}

	template<typename TB, typename TE, typename TJ>
	inline void ScatterJ(TB const & pB, TE const &pE, TJ & pJ) const
	{
		ForEachCell(

		[&pB,&pE,&pJ](cell_type & cell,index_type const &hint_idx)
		{
			auto fB = MakeCache(pB, hint_idx);
			auto fE = MakeCache(pE, hint_idx);
			auto fJ = MakeCache(pJ, hint_idx);

			for (auto const & p : cell)
			{
				fJ.Scatter(p.x, p.v);
			}
		});

		mesh_->UpdateBoundary(pJ);

	}

}
;

} // namespace simpla

#endif /* GGAUGE_H_ */
