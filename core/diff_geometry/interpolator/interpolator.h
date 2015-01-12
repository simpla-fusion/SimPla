/**
 * @file  interpolator.h
 *
 *  created on: 2014-4-17
 *      Author: salmon
 */

#ifndef INTERPOLATOR_H_
#define INTERPOLATOR_H_

//#include "../../fetl/field_constant.h"
#include "../../gtl/ntuple.h"
#include "../../utilities/primitives.h"

namespace simpla
{

template<typename ...> class field_traits;

/**
 * @ingroup diff_geo
 * @addtogroup interpolator Interpolator
 * @brief   mapping discrete points to continue space
 *
 */
/**
 * @ingroup interpolator
 * @brief basic linear interpolator
 */
template<typename G>
class InterpolatorLinear
{

public:
	typedef InterpolatorLinear<G> this_type;

	typedef G geometry_type;

	typedef typename G::coordinates_type coordinates_type;
	typedef typename G::topology_type topology_type;

	G const * geo;
	InterpolatorLinear(G const * g) :
			geo(g)
	{

	}
	InterpolatorLinear() :
			geo(nullptr)
	{

	}
	InterpolatorLinear(this_type const & r) :
			geo(r.geo)
	{

	}
	~InterpolatorLinear()
	{

	}

	void geometry(G const*g)
	{
		geo = g;
	}
	G const &geometry() const
	{
		return *geo;
	}

//	template<typename ... Args>
//	auto gather(Args && ...args) const
//	DECL_RET_TYPE ((gather_(std::forward<Args> (args)...)))
//
//	template<typename ... Args>
//	auto scatter(Args && ...args) const
//	DECL_RET_TYPE (scatter_(std::forward<Args> (args)...))

private:

	template<typename TD, typename TIDX>
	inline auto gather_impl_(TD const & f,
			TIDX const & idx) const -> decltype(get_value(f, std::get<0>(idx) )* std::get<1>(idx)[0])
	{

		auto X = (topology_type::_DI) << 1;
		auto Y = (topology_type::_DJ) << 1;
		auto Z = (topology_type::_DK) << 1;

		typename G::coordinates_type r = std::get<1>(idx);
		typename G::index_type s = std::get<0>(idx);

		return get_value(f, ((s + X) + Y) + Z) * (r[0]) * (r[1]) * (r[2]) //
		+ get_value(f, (s + X) + Y) * (r[0]) * (r[1]) * (1.0 - r[2]) //
		+ get_value(f, (s + X) + Z) * (r[0]) * (1.0 - r[1]) * (r[2]) //
		+ get_value(f, (s + X)) * (r[0]) * (1.0 - r[1]) * (1.0 - r[2]) //
		+ get_value(f, (s + Y) + Z) * (1.0 - r[0]) * (r[1]) * (r[2]) //
		+ get_value(f, (s + Y)) * (1.0 - r[0]) * (r[1]) * (1.0 - r[2]) //
		+ get_value(f, s + Z) * (1.0 - r[0]) * (1.0 - r[1]) * (r[2]) //
		+ get_value(f, s) * (1.0 - r[0]) * (1.0 - r[1]) * (1.0 - r[2]);
	}
public:

	template<typename TF>
	inline auto gather(TF const &f,
			coordinates_type const & r) const //
					ENABLE_IF_DECL_RET_TYPE((field_traits<TF >::iform==VERTEX),
							( gather_impl_(f, geo->coordinates_global_to_local(r, 0UL) )))

	template<typename TF>
	auto gather(TF const &f,
			coordinates_type const & r) const
					ENABLE_IF_DECL_RET_TYPE((field_traits<TF >::iform==EDGE),
							make_nTuple(
									gather_impl_(f, geo->coordinates_global_to_local(r, (topology_type::_DI)) ),
									gather_impl_(f, geo->coordinates_global_to_local(r, (topology_type::_DJ)) ),
									gather_impl_(f, geo->coordinates_global_to_local(r, (topology_type::_DK)) )
							))

	template<typename TF>
	auto gather(TF const &f,
			coordinates_type const & r) const
					ENABLE_IF_DECL_RET_TYPE(
							(field_traits<TF >::iform==FACE),
							make_nTuple(
									gather_impl_(f, geo->coordinates_global_to_local(r,((topology_type::_DJ | topology_type::_DK))) ),
									gather_impl_(f, geo->coordinates_global_to_local(r,((topology_type::_DK | topology_type::_DI))) ),
									gather_impl_(f, geo->coordinates_global_to_local(r,((topology_type::_DI | topology_type::_DJ))) )
							) )

	template<typename TF>
	auto gather(TF const &f,
			coordinates_type const & x) const
					ENABLE_IF_DECL_RET_TYPE((field_traits<TF >::iform==VOLUME),
							gather_impl_(f, geo->coordinates_global_to_local(x, (topology_type::_DA)) ))

private:
	template<typename TF, typename IDX, typename TV>
	inline void scatter_impl_(TF &f, IDX const& idx, TV const & v) const
	{

		auto X = (topology_type::_DI) << 1;
		auto Y = (topology_type::_DJ) << 1;
		auto Z = (topology_type::_DK) << 1;

		typename G::coordinates_type r = std::get<1>(idx);
		typename G::index_type s = std::get<0>(idx);

		get_value(f, ((s + X) + Y) + Z) += v * (r[0]) * (r[1]) * (r[2]);
		get_value(f, (s + X) + Y) += v * (r[0]) * (r[1]) * (1.0 - r[2]);
		get_value(f, (s + X) + Z) += v * (r[0]) * (1.0 - r[1]) * (r[2]);
		get_value(f, s + X) += v * (r[0]) * (1.0 - r[1]) * (1.0 - r[2]);
		get_value(f, (s + Y) + Z) += v * (1.0 - r[0]) * (r[1]) * (r[2]);
		get_value(f, s + Y) += v * (1.0 - r[0]) * (r[1]) * (1.0 - r[2]);
		get_value(f, s + Z) += v * (1.0 - r[0]) * (1.0 - r[1]) * (r[2]);
		get_value(f, s) += v * (1.0 - r[0]) * (1.0 - r[1]) * (1.0 - r[2]);
	}
public:

	template<typename TF, typename TV, typename TW>
	auto scatter(TF &f, coordinates_type const & x, TV const &u,
			TW const &w) const ->typename std::enable_if< (field_traits<TF >::iform==VERTEX)>::type
	{

		scatter_impl_(f, geo->coordinates_global_to_local(x, 0UL), u * w);
	}

	template<typename TF, typename TV, typename TW>
	auto scatter(TF &f, coordinates_type const & x, TV const &u,
			TW const & w) const ->typename std::enable_if< (field_traits<TF >::iform==EDGE)>::type
	{
		scatter_impl_(f,
				geo->coordinates_global_to_local(x, (topology_type::_DI)),
				u[0] * w);
		scatter_impl_(f,
				geo->coordinates_global_to_local(x, (topology_type::_DJ)),
				u[1] * w);
		scatter_impl_(f,
				geo->coordinates_global_to_local(x, (topology_type::_DK)),
				u[2] * w);

	}

	template<typename TF, typename TV, typename TW>
	auto scatter(TF &f, coordinates_type const & x, TV const &u,
			TW const &w) const ->typename std::enable_if< (field_traits<TF >::iform==FACE)>::type
	{

		scatter_impl_(f,
				geo->coordinates_global_to_local(x,
						((topology_type::_DJ | topology_type::_DK))), u[0] * w);
		scatter_impl_(f,
				geo->coordinates_global_to_local(x,
						((topology_type::_DK | topology_type::_DI))), u[1] * w);
		scatter_impl_(f,
				geo->coordinates_global_to_local(x,
						((topology_type::_DI | topology_type::_DJ))), u[2] * w);
	}

	template<typename TF, typename TV, typename TW>
	auto scatter(TF &f, coordinates_type const & x, TV const &u,
			TW const &w) const ->typename std::enable_if< (field_traits<TF >::iform==VOLUME)>::type
	{
		scatter_impl_(f,
				geo->coordinates_global_to_local(x, topology_type::_DA), w);
	}

}
;

}
// namespace simpla

#endif /* INTERPOLATOR_H_ */
