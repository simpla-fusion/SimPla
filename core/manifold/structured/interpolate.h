/**
 * @file  interpolate.h
 *
 *  created on: 2014-4-17
 *      Author: salmon
 */

#ifndef INTERPOLATE_H_
#define INTERPOLATE_H_

#include <stddef.h>
#include <type_traits>

#include "../../gtl/type_traits.h"
#include "../../gtl/ntuple.h"
#include "manifold_traits.h"

namespace simpla
{

template<typename ...> class Field;

template<typename ...> class Domain;

namespace tags
{
struct linear;
}

/**
 * @ingroup diff_geo
 * @addtogroup interpolate Interpolate
 * @brief   mapping discrete points to continue space
 *
 */
/**
 * @ingroup interpolate
 * @brief basic linear interpolate
 */
namespace interpolate
{
template<typename TM, typename TAGS> struct Interpolate;

template<typename TM>
struct Interpolate<TM, tags::linear>
{

public:

	typedef TM mesh_type;

	typedef typename mesh_type::id_type id;

	typedef Interpolate<mesh_type, tags::linear> this_type;

private:

	template<typename TD, typename TIDX>
	static auto gather_impl_(TD const &f,
			TIDX const &idx) -> decltype(traits::index(f, std::get<0>(idx)) * std::get<1>(idx)[0])
	{

		auto X = (mesh_type::_DI) << 1;
		auto Y = (mesh_type::_DJ) << 1;
		auto Z = (mesh_type::_DK) << 1;

		typename mesh_type::point_type r = std::get<1>(idx);
		typename mesh_type::index_type s = std::get<0>(idx);

		return traits::index(f, ((s + X) + Y) + Z) * (r[0]) * (r[1]) * (r[2]) //
				+ traits::index(f, (s + X) + Y) * (r[0]) * (r[1]) * (1.0 - r[2]) //
				+ traits::index(f, (s + X) + Z) * (r[0]) * (1.0 - r[1]) * (r[2]) //
				+ traits::index(f, (s + X)) * (r[0]) * (1.0 - r[1]) * (1.0 - r[2]) //
				+ traits::index(f, (s + Y) + Z) * (1.0 - r[0]) * (r[1]) * (r[2]) //
				+ traits::index(f, (s + Y)) * (1.0 - r[0]) * (r[1]) * (1.0 - r[2]) //
				+ traits::index(f, s + Z) * (1.0 - r[0]) * (1.0 - r[1]) * (r[2]) //
				+ traits::index(f, s) * (1.0 - r[0]) * (1.0 - r[1]) * (1.0 - r[2]);
	}

public:

	template<typename TF, typename TX>
	static inline auto gather(mesh_type const &geo, TF const &f,
			TX const &r)  //
	ENABLE_IF_DECL_RET_TYPE((traits::iform<TF>::value == VERTEX),
			(gather_impl_(f, geo.coordinates_global_to_local(r, 0))))

	template<typename TF>
	static auto gather(mesh_type const &geo, TF const &f,
			typename mesh_type::point_type const &r)
	ENABLE_IF_DECL_RET_TYPE((traits::iform<TF>::value == EDGE),
			make_nTuple(
					gather_impl_(f, geo.coordinates_global_to_local(r, 1)),
					gather_impl_(f, geo.coordinates_global_to_local(r, 2)),
					gather_impl_(f, geo.coordinates_global_to_local(r, 4))
			))

	template<typename TF>
	static auto gather(mesh_type const &geo, TF const &f,
			typename mesh_type::point_type const &r)
	ENABLE_IF_DECL_RET_TYPE(
			(traits::iform<TF>::value == FACE),
			make_nTuple(
					gather_impl_(f, geo.coordinates_global_to_local(r, 6)),
					gather_impl_(f, geo.coordinates_global_to_local(r, 5)),
					gather_impl_(f, geo.coordinates_global_to_local(r, 3))
			))

	template<typename TF>
	static auto gather(mesh_type const &geo, TF const &f,
			typename mesh_type::point_type const &x)
	ENABLE_IF_DECL_RET_TYPE((traits::iform<TF>::value == VOLUME),
			gather_impl_(f, geo.coordinates_global_to_local(x, 7)))

private:
	template<typename TF, typename IDX, typename TV>
	static inline void scatter_impl_(TF &f, IDX const &idx, TV const &v)
	{

		auto X = (mesh_type::_DI) << 1;
		auto Y = (mesh_type::_DJ) << 1;
		auto Z = (mesh_type::_DK) << 1;

		typename mesh_type::point_type r = std::get<1>(idx);
		typename mesh_type::index_type s = std::get<0>(idx);

		traits::index(f, ((s + X) + Y) + Z) += v * (r[0]) * (r[1]) * (r[2]);
		traits::index(f, (s + X) + Y) += v * (r[0]) * (r[1]) * (1.0 - r[2]);
		traits::index(f, (s + X) + Z) += v * (r[0]) * (1.0 - r[1]) * (r[2]);
		traits::index(f, s + X) += v * (r[0]) * (1.0 - r[1]) * (1.0 - r[2]);
		traits::index(f, (s + Y) + Z) += v * (1.0 - r[0]) * (r[1]) * (r[2]);
		traits::index(f, s + Y) += v * (1.0 - r[0]) * (r[1]) * (1.0 - r[2]);
		traits::index(f, s + Z) += v * (1.0 - r[0]) * (1.0 - r[1]) * (r[2]);
		traits::index(f, s) += v * (1.0 - r[0]) * (1.0 - r[1]) * (1.0 - r[2]);

	}

public:

	template<typename ...Others, typename ...TF, typename TV, typename TW>
	static void scatter(mesh_type const &geo,
			Field<
					Domain<mesh_type, std::integral_constant<int, VERTEX>,
							Others...>, TF...> &f,
			typename mesh_type::point_type const &x, TV const &u, TW const &w)
	{

		scatter_impl_(f, geo.coordinates_global_to_local(x, 0), u * w);
	}

	template<typename ...Others, typename ...TF, typename TV, typename TW>
	static void scatter(mesh_type const &geo,
			Field<
					Domain<mesh_type, std::integral_constant<int, EDGE>,
							Others...>, TF...> &f,
			typename mesh_type::point_type const &x, TV const &u, TW const &w)
	{

		scatter_impl_(f, geo.coordinates_global_to_local(x, 1), u[0] * w);
		scatter_impl_(f, geo.coordinates_global_to_local(x, 2), u[1] * w);
		scatter_impl_(f, geo.coordinates_global_to_local(x, 4), u[2] * w);

	}

	template<typename ...Others, typename ...TF, typename TV, typename TW>
	static void scatter(mesh_type const &geo,
			Field<
					Domain<mesh_type, std::integral_constant<int, FACE>,
							Others...>, TF...> &f,
			typename mesh_type::point_type const &x, TV const &u, TW const &w)
	{

		scatter_impl_(f, geo.coordinates_global_to_local(x, 6), u[0] * w);
		scatter_impl_(f, geo.coordinates_global_to_local(x, 5), u[1] * w);
		scatter_impl_(f, geo.coordinates_global_to_local(x, 3), u[2] * w);
	}

	template<typename ...Others, typename ...TF, typename TV, typename TW>
	static void scatter(mesh_type const &geo,
			Field<
					Domain<mesh_type, std::integral_constant<int, VOLUME>,
							Others...>, TF...> &f,
			typename mesh_type::point_type const &x, TV const &u, TW const &w)
	{
		scatter_impl_(f, geo.coordinates_global_to_local(x, 7), w);
	}

private:
	template<typename TV>
	static TV sample_(mesh_type const &geo,
			std::integral_constant<int, VERTEX>, id s, TV const &v)
	{
		return v;
	}

	template<typename TV>
	static TV sample_(mesh_type const &geo,
			std::integral_constant<int, VOLUME>, id s, TV const &v)
	{
		return v;
	}

	template<typename TV>
	static TV sample_(mesh_type const &geo, std::integral_constant<int, EDGE>,
			id s, nTuple<TV, 3> const &v)
	{
		return v[mesh_type::sub_index(s)];
	}

	template<typename TV>
	static TV sample_(mesh_type const &geo, std::integral_constant<int, FACE>,
			id s, nTuple<TV, 3> const &v)
	{
		return v[mesh_type::sub_index(s)];
	}

	template<int IFORM, typename TV>
	static TV sample_(mesh_type const &geo, std::integral_constant<int, IFORM>,
			id s, TV const &v)
	{
		return v;
	}

public:

	template<int IFORM, typename ...Args>
	static auto sample(mesh_type const &geo, Args &&... args)
	DECL_RET_TYPE((sample_(geo, std::integral_constant<int, IFORM>(),
			std::forward<Args>(args)...)))
};
}  //namespace solver
}
// namespace simpla

#endif /* INTERPOLATE_H_ */
