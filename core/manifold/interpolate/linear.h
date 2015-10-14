/**
 * @file linear.h
 * @author salmon
 * @date 2015-10-14.
 */

#ifndef SIMPLA_LINEAR_H
#define SIMPLA_LINEAR_H

#include "interpolate.h"

namespace simpla
{


/**
 * @ingroup diff_geo
 * @addtogroup interpolate Interpolate
 * @brief   mapping discrete points to continue space
 *
 */

namespace pt= interpolate::tags;

#define DECLARE_FUNCTION_SUFFIX const
#define DECLARE_FUNCTION_PREFIX

/**
 * @ingroup interpolate
 * @brief basic linear interpolate
 */
template<typename TGeo>
struct Interpolate<TGeo, pt::linear>
{
private:
	
	typedef TGeo geometry_type;
	
	typedef typename geometry_type::id_type id;
	
	typedef Interpolate<geometry_type, pt::linear> this_type;

	geometry_type const &m_geo_;

private:
	
	template<typename TD, typename TIDX>
	DECLARE_FUNCTION_PREFIX auto gather_impl_(TD const &f,
			TIDX const &idx) DECLARE_FUNCTION_SUFFIX -> decltype(traits::index(f, std::get<0>(idx)) *
			std::get<1>(idx)[0])
	{
		
		auto X = (geometry_type::_DI) << 1;
		auto Y = (geometry_type::_DJ) << 1;
		auto Z = (geometry_type::_DK) << 1;
		
		typename geometry_type::point_type r = std::get<1>(idx);
		typename geometry_type::index_type s = std::get<0>(idx);
		
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
	DECLARE_FUNCTION_PREFIX auto gather(TF const &f,
			TX const &r) DECLARE_FUNCTION_SUFFIX//
	ENABLE_IF_DECL_RET_TYPE((traits::iform<TF>::value
			== VERTEX),
			(
					gather_impl_(f, m_geo_.coordinates_global_to_local(r,
							0))))
	
	template<typename TF>
	DECLARE_FUNCTION_PREFIX auto gather(TF const &f,
			typename geometry_type::point_type const &r) DECLARE_FUNCTION_SUFFIX
	ENABLE_IF_DECL_RET_TYPE((traits::iform<TF>::value
			== EDGE),
			
			make_nTuple(
					gather_impl_(f, m_geo_.coordinates_global_to_local(r, 1)),
					gather_impl_(f, m_geo_.coordinates_global_to_local(r, 2)),
					gather_impl_(f, m_geo_.coordinates_global_to_local(r, 4))
			))
	
	template<typename TF>
	DECLARE_FUNCTION_PREFIX auto gather(TF const &f,
			typename geometry_type::point_type const &r) DECLARE_FUNCTION_SUFFIX
	ENABLE_IF_DECL_RET_TYPE((traits::iform<TF>::value
			== FACE),
			
			make_nTuple(
					gather_impl_(f, m_geo_.coordinates_global_to_local(r, 6)),
					gather_impl_(f, m_geo_.coordinates_global_to_local(r, 5)),
					gather_impl_(f, m_geo_.coordinates_global_to_local(r, 3))
			))
	
	template<typename TF>
	DECLARE_FUNCTION_PREFIX auto gather(TF const &f,
			typename geometry_type::point_type const &x) DECLARE_FUNCTION_SUFFIX
	ENABLE_IF_DECL_RET_TYPE((traits::iform<TF>::value
			== VOLUME),
			gather_impl_(f, m_geo_
					.
							coordinates_global_to_local(x,
							7)))

private:
	template<typename TF, typename IDX, typename TV>
	DECLARE_FUNCTION_PREFIX void scatter_impl_(TF &f, IDX const &idx, TV const &v) DECLARE_FUNCTION_SUFFIX
	{
		
		auto X = (geometry_type::_DI) << 1;
		auto Y = (geometry_type::_DJ) << 1;
		auto Z = (geometry_type::_DK) << 1;
		
		typename geometry_type::point_type r = std::get<1>(idx);
		typename geometry_type::index_type s = std::get<0>(idx);
		
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
	DECLARE_FUNCTION_PREFIX void scatter(Field<Domain<geometry_type, std::integral_constant<int, VERTEX>,
			Others...>, TF...> &f,
			typename geometry_type::point_type const &x, TV const &u, TW const &w) DECLARE_FUNCTION_SUFFIX
	{
		scatter_impl_(f, m_geo_.coordinates_global_to_local(x, 0), u * w);
	}
	
	template<typename ...Others, typename ...TF, typename TV, typename TW>
	DECLARE_FUNCTION_PREFIX void scatter(Field<Domain<geometry_type, std::integral_constant<int, EDGE>, Others...
	>, TF...> &f, typename geometry_type::point_type const &x, TV const &u, TW const &w
	) DECLARE_FUNCTION_SUFFIX
	{
		
		scatter_impl_(f, m_geo_.coordinates_global_to_local(x, 1), u[0] * w);
		scatter_impl_(f, m_geo_.coordinates_global_to_local(x, 2), u[1] * w);
		scatter_impl_(f, m_geo_.coordinates_global_to_local(x, 4), u[2] * w);
		
	}
	
	template<typename ...Others, typename ...TF, typename TV, typename TW>
	DECLARE_FUNCTION_PREFIX void scatter(Field<Domain<geometry_type, std::integral_constant<int, FACE>,
			Others...>, TF...> &f, typename geometry_type::point_type const &x, TV const &u,
			TW const &w) DECLARE_FUNCTION_SUFFIX
	{
		
		scatter_impl_(f, m_geo_.coordinates_global_to_local(x, 6), u[0] * w);
		scatter_impl_(f, m_geo_.coordinates_global_to_local(x, 5), u[1] * w);
		scatter_impl_(f, m_geo_.coordinates_global_to_local(x, 3), u[2] * w);
	}
	
	template<typename ...Others, typename ...TF, typename TV, typename TW>
	DECLARE_FUNCTION_PREFIX void scatter(
			Field<Domain<geometry_type, std::integral_constant<int, VOLUME>,
					Others...>, TF...> &f, typename geometry_type::point_type const &x, TV const &u, TW const &w
	) DECLARE_FUNCTION_SUFFIX
	{
		scatter_impl_(f, m_geo_.coordinates_global_to_local(x, 7), w);
	}

private:
	template<typename TV>
	DECLARE_FUNCTION_PREFIX TV sample_(std::integral_constant<int, VERTEX>, id s,
			TV const &v) DECLARE_FUNCTION_SUFFIX { return v; }
	
	template<typename TV>
	DECLARE_FUNCTION_PREFIX TV sample_(std::integral_constant<int, VOLUME>, id s,
			TV const &v) DECLARE_FUNCTION_SUFFIX { return v; }
	
	template<typename TV>
	DECLARE_FUNCTION_PREFIX TV sample_(std::integral_constant<int, EDGE>,
			id s, nTuple<TV, 3> const &v) DECLARE_FUNCTION_SUFFIX { return v[geometry_type::sub_index(s)]; }
	
	template<typename TV>
	DECLARE_FUNCTION_PREFIX TV sample_(std::integral_constant<int, FACE>,
			id s, nTuple<TV, 3> const &v) DECLARE_FUNCTION_SUFFIX { return v[geometry_type::sub_index(s)]; }
	
	template<int IFORM, typename TV>
	DECLARE_FUNCTION_PREFIX TV sample_(std::integral_constant<int, IFORM>, id s,
			TV const &v) DECLARE_FUNCTION_SUFFIX { return v; }

public:
	
	template<int IFORM, typename ...Args>
	DECLARE_FUNCTION_PREFIX auto sample(Args &&... args) DECLARE_FUNCTION_SUFFIX
	DECL_RET_TYPE((sample_(m_geo_, std::integral_constant<int, IFORM>(),
			std::forward<Args>(args)...)))


public:
	Interpolate(geometry_type &geo) : m_geo_(geo)
	{
	}

	virtual ~Interpolate()
	{
	}

	
};
	
	
}//namespace simpla
#endif //SIMPLA_LINEAR_H