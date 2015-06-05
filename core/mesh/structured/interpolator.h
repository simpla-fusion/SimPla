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
namespace simpla
{

template<typename ...> class _Field;
template<typename, size_t> class Domain;

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
			TIDX const & idx) -> decltype(try_index(f, std::get<0>(idx) )* std::get<1>(idx)[0])
	{

		auto X = (geometry_type::_DI) << 1;
		auto Y = (geometry_type::_DJ) << 1;
		auto Z = (geometry_type::_DK) << 1;

		typename geometry_type::point_type r = std::get<1>(idx);
		typename geometry_type::index_type s = std::get<0>(idx);

		return try_index(f, ((s + X) + Y) + Z) * (r[0]) * (r[1]) * (r[2]) //
		+ try_index(f, (s + X) + Y) * (r[0]) * (r[1]) * (1.0 - r[2]) //
		+ try_index(f, (s + X) + Z) * (r[0]) * (1.0 - r[1]) * (r[2]) //
		+ try_index(f, (s + X)) * (r[0]) * (1.0 - r[1]) * (1.0 - r[2]) //
		+ try_index(f, (s + Y) + Z) * (1.0 - r[0]) * (r[1]) * (r[2]) //
		+ try_index(f, (s + Y)) * (1.0 - r[0]) * (r[1]) * (1.0 - r[2]) //
		+ try_index(f, s + Z) * (1.0 - r[0]) * (1.0 - r[1]) * (r[2]) //
		+ try_index(f, s) * (1.0 - r[0]) * (1.0 - r[1]) * (1.0 - r[2]);
	}
public:

	template<typename geometry_type, typename TF, typename TX>
	static inline auto gather(geometry_type const & geo, TF const &f,
			TX const & r)  //
					ENABLE_IF_DECL_RET_TYPE((field_traits<TF>::iform==VERTEX),
							( gather_impl_<geometry_type>(f, geo. coordinates_global_to_local (r, 0 ) )))

	template<typename geometry_type, typename TF>
	static auto gather(geometry_type const & geo, TF const &f,
			typename geometry_type::point_type const & r)
					ENABLE_IF_DECL_RET_TYPE((field_traits<TF >::iform==EDGE),
							make_nTuple(
									gather_impl_<geometry_type>(f, geo.coordinates_global_to_local(r, 1) ),
									gather_impl_<geometry_type>(f, geo.coordinates_global_to_local(r, 2) ),
									gather_impl_<geometry_type>(f, geo.coordinates_global_to_local(r, 4) )
							))

	template<typename geometry_type, typename TF>
	static auto gather(geometry_type const & geo, TF const &f,
			typename geometry_type::point_type const & r)
					ENABLE_IF_DECL_RET_TYPE(
							(field_traits<TF >::iform==FACE),
							make_nTuple(
									gather_impl_<geometry_type>(f, geo.coordinates_global_to_local(r,6) ),
									gather_impl_<geometry_type>(f, geo.coordinates_global_to_local(r,5) ),
									gather_impl_<geometry_type>(f, geo.coordinates_global_to_local(r,3) )
							) )

	template<typename geometry_type, typename TF>
	static auto gather(geometry_type const & geo, TF const &f,
			typename geometry_type::point_type const & x)
					ENABLE_IF_DECL_RET_TYPE((field_traits<TF >::iform==VOLUME),
							gather_impl_<geometry_type>(f, geo. coordinates_global_to_local (x ,7) ))

private:
	template<typename geometry_type, typename TF, typename IDX, typename TV>
	static inline void scatter_impl_(TF &f, IDX const& idx, TV const & v)
	{

		auto X = (geometry_type::_DI) << 1;
		auto Y = (geometry_type::_DJ) << 1;
		auto Z = (geometry_type::_DK) << 1;

		typename geometry_type::point_type r = std::get<1>(idx);
		typename geometry_type::index_type s = std::get<0>(idx);

		try_index(f, ((s + X) + Y) + Z) += v * (r[0]) * (r[1]) * (r[2]);
		try_index(f, (s + X) + Y) += v * (r[0]) * (r[1]) * (1.0 - r[2]);
		try_index(f, (s + X) + Z) += v * (r[0]) * (1.0 - r[1]) * (r[2]);
		try_index(f, s + X) += v * (r[0]) * (1.0 - r[1]) * (1.0 - r[2]);
		try_index(f, (s + Y) + Z) += v * (1.0 - r[0]) * (r[1]) * (r[2]);
		try_index(f, s + Y) += v * (1.0 - r[0]) * (r[1]) * (1.0 - r[2]);
		try_index(f, s + Z) += v * (1.0 - r[0]) * (1.0 - r[1]) * (r[2]);
		try_index(f, s) += v * (1.0 - r[0]) * (1.0 - r[1]) * (1.0 - r[2]);

	}
public:

	template<typename geometry_type, typename ...TF, typename TV, typename TW>
	static void scatter(geometry_type const & geo,
			_Field<Domain<geometry_type, VERTEX>, TF...> &f,
			typename geometry_type::point_type const & x, TV const &u,
			TW const &w)
	{

		scatter_impl_<geometry_type>(f, geo.coordinates_global_to_local(x, 0),
				u * w);
	}

	template<typename geometry_type, typename ...TF, typename TV, typename TW>
	static void scatter(geometry_type const & geo,
			_Field<Domain<geometry_type, EDGE>, TF...> &f,
			typename geometry_type::point_type const & x, TV const &u,
			TW const & w)
	{

		scatter_impl_<geometry_type>(f, geo.coordinates_global_to_local(x, 1),
				u[0] * w);
		scatter_impl_<geometry_type>(f, geo.coordinates_global_to_local(x, 2),
				u[1] * w);
		scatter_impl_<geometry_type>(f, geo.coordinates_global_to_local(x, 4),
				u[2] * w);

	}

	template<typename geometry_type, typename ...TF, typename TV, typename TW>
	static void scatter(geometry_type const & geo,
			_Field<Domain<geometry_type, FACE>, TF...>&f,
			typename geometry_type::point_type const & x, TV const &u,
			TW const &w)
	{

		scatter_impl_<geometry_type>(f, geo.coordinates_global_to_local(x, 6),
				u[0] * w);
		scatter_impl_<geometry_type>(f, geo.coordinates_global_to_local(x, 5),
				u[1] * w);
		scatter_impl_<geometry_type>(f, geo.coordinates_global_to_local(x, 3),
				u[2] * w);
	}

	template<typename geometry_type, typename ...TF, typename TV, typename TW>
	static void scatter(geometry_type const & geo,
			_Field<Domain<geometry_type, VOLUME>, TF...>&f,
			typename geometry_type::point_type const & x, TV const &u,
			TW const &w)
	{
		scatter_impl_<geometry_type>(f, geo.coordinates_global_to_local(x, 7),
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
