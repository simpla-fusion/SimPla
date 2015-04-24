/**
 * @file  interpolator.h
 *
 *  created on: 2014-4-17
 *      Author: salmon
 */

#ifndef INTERPOLATOR_H_
#define INTERPOLATOR_H_

#include <stddef.h>
#include <type_traits>

#include "../../gtl/type_traits.h"
#include "../../gtl/ntuple.h"
#include "../mesh.h"
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
class InterpolatorLinear
{

public:
	typedef InterpolatorLinear this_type;

	InterpolatorLinear()
	{
	}

	InterpolatorLinear(this_type const & r) = default;

	~InterpolatorLinear() = default;

private:

	template<typename geometry_type, typename TD, typename TIDX>
	static auto gather_impl_(TD const & f,
			TIDX const & idx) -> decltype(get_value(f, std::get<0>(idx) )* std::get<1>(idx)[0])
	{

		auto X = (geometry_type::_DI) << 1;
		auto Y = (geometry_type::_DJ) << 1;
		auto Z = (geometry_type::_DK) << 1;

		typename geometry_type::coordinates_type r = std::get<1>(idx);
		typename geometry_type::index_type s = std::get<0>(idx);

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

	template<typename geometry_type, typename TF, typename TX>
	static inline auto gather(geometry_type const & geo, TF const &f,
			TX const & r)  //
					ENABLE_IF_DECL_RET_TYPE((field_traits<TF >::iform==VERTEX),
							( gather_impl_<geometry_type>(f, geo.template coordinates_global_to_local<VERTEX>(r, 0 ) )))

	template<typename geometry_type, typename TF>
	static auto gather(geometry_type const & geo, TF const &f,
			typename geometry_type::coordinates_type const & r)
					ENABLE_IF_DECL_RET_TYPE((field_traits<TF >::iform==EDGE),
							make_nTuple(
									gather_impl_<geometry_type>(f, geo.template coordinates_global_to_local<EDGE>(r, 0) ),
									gather_impl_<geometry_type>(f, geo.template coordinates_global_to_local<EDGE>(r, 1) ),
									gather_impl_<geometry_type>(f, geo.template coordinates_global_to_local<EDGE>(r, 2) )
							))

	template<typename geometry_type, typename TF>
	static auto gather(geometry_type const & geo, TF const &f,
			typename geometry_type::coordinates_type const & r)
					ENABLE_IF_DECL_RET_TYPE(
							(field_traits<TF >::iform==FACE),
							make_nTuple(
									gather_impl_<geometry_type>(f, geo.template coordinates_global_to_local<FACE>(r,0) ),
									gather_impl_<geometry_type>(f, geo.template coordinates_global_to_local<FACE>(r,1) ),
									gather_impl_<geometry_type>(f, geo.template coordinates_global_to_local<FACE>(r,2) )
							) )

	template<typename geometry_type, typename TF>
	static auto gather(geometry_type const & geo, TF const &f,
			typename geometry_type::coordinates_type const & x)
					ENABLE_IF_DECL_RET_TYPE((field_traits<TF >::iform==VOLUME),
							gather_impl_<geometry_type>(f, geo.template coordinates_global_to_local<VOLUME>(x ) ))

private:
	template<typename geometry_type, typename TF, typename IDX, typename TV>
	static inline void scatter_impl_(TF &f, IDX const& idx, TV const & v)
	{

		auto X = (geometry_type::_DI) << 1;
		auto Y = (geometry_type::_DJ) << 1;
		auto Z = (geometry_type::_DK) << 1;

		typename geometry_type::coordinates_type r = std::get<1>(idx);
		typename geometry_type::index_type s = std::get<0>(idx);

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

	template<typename geometry_type, typename TF, typename TV, typename TW>
	static auto scatter(geometry_type const & geo, TF &f,
			typename geometry_type::coordinates_type const & x, TV const &u,
			TW const &w) ->typename std::enable_if< (field_traits<TF >::iform==VERTEX)>::type
	{

		scatter_impl_(f, geo.coordinates_global_to_local<VERTEX>(x), u * w);
	}

	template<typename geometry_type, typename TF, typename TV, typename TW>
	static auto scatter(geometry_type const & geo, TF &f,
			typename geometry_type::coordinates_type const & x, TV const &u,
			TW const & w) ->typename std::enable_if< (field_traits<TF >::iform==EDGE)>::type
	{
		scatter_impl_(f, geo.coordinates_global_to_local<EDGE>(x, 0), u[0] * w);
		scatter_impl_(f, geo.coordinates_global_to_local<EDGE>(x, 1), u[1] * w);
		scatter_impl_(f, geo.coordinates_global_to_local<EDGE>(x, 2), u[2] * w);

	}

	template<typename geometry_type, typename TF, typename TV, typename TW>
	static auto scatter(geometry_type const & geo, TF &f,
			typename geometry_type::coordinates_type const & x, TV const &u,
			TW const &w) ->typename std::enable_if< (field_traits<TF >::iform==FACE)>::type
	{

		scatter_impl_(f, geo.coordinates_global_to_local<FACE>(x, 0), u[0] * w);
		scatter_impl_(f, geo.coordinates_global_to_local<FACE>(x, 1), u[1] * w);
		scatter_impl_(f, geo.coordinates_global_to_local<FACE>(x, 2), u[2] * w);
	}

	template<typename geometry_type, typename TF, typename TV, typename TW>
	static auto scatter(geometry_type const & geo, TF &f,
			typename geometry_type::coordinates_type const & x, TV const &u,
			TW const &w) ->typename std::enable_if< (field_traits<TF >::iform==VOLUME)>::type
	{
		scatter_impl_(f, geo.coordinates_global_to_local(x, geometry_type::_DA),
				w);
	}
private:
	template<typename geometry_type, typename TV>
	static TV sample_(geometry_type const & geo,
			std::integral_constant<size_t, VERTEX>, size_t s, TV const &v)
	{
		return v;
	}

	template<typename geometry_type, typename TV>
	static TV sample_(geometry_type const & geo,
			std::integral_constant<size_t, VOLUME>, size_t s, TV const &v)
	{
		return v;
	}

	template<typename geometry_type, typename TV>
	static TV sample_(geometry_type const & geo,
			std::integral_constant<size_t, EDGE>, size_t s,
			nTuple<TV, 3> const &v)
	{
		return v[geometry_type::sub_index(s)];
	}

	template<typename geometry_type, typename TV>
	static TV sample_(geometry_type const & geo,
			std::integral_constant<size_t, FACE>, size_t s,
			nTuple<TV, 3> const &v)
	{
		return v[geometry_type::sub_index(s)];
	}

	template<typename geometry_type, size_t IFORM, typename TV>
	static TV sample_(geometry_type const & geo,
			std::integral_constant<size_t, IFORM>, size_t s, TV const & v)
	{
		return v;
	}
public:

	template<size_t IFORM, typename geometry_type, typename ...Args>
	static auto sample(geometry_type const & geo, Args && ... args)
	DECL_RET_TYPE((sample_(geo,std::integral_constant<size_t, IFORM>(),
							std::forward<Args>(args)...)))
}
;

}
// namespace simpla

#endif /* INTERPOLATOR_H_ */
