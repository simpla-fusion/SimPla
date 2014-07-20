/*
 * interpolator.h
 *
 *  created on: 2014-4-17
 *      Author: salmon
 */

#ifndef INTERPOLATOR_H_
#define INTERPOLATOR_H_

#include "../utilities/primitives.h"

namespace simpla
{

template<typename, unsigned int, typename > class Field;
/**
 * \ingroup Mesh
 *
 * \brief Interpolator
 */
template<typename TM, typename Policy = std::nullptr_t>
class Interpolator
{
public:

	typedef TM mesh_type;
	typedef typename mesh_type::topology_type topology_type;
	typedef typename mesh_type::geometry_type geometry_type;
	typedef typename topology_type::coordinates_type coordinates_type;
	typedef typename topology_type::iterator iterator;
	typedef typename topology_type::compact_index_type compact_index_type;
	typedef typename mesh_type::scalar_type scalar_type;

private:
	template<typename TF, typename TIDX>
	static inline auto Gather_impl_(TF const & f, TIDX idx)
	->decltype(get_value(f, std::get<0>(idx) )* std::get<1>(idx)[0])
	{

		auto X = (topology_type::_DI) << 1;
		auto Y = (topology_type::_DJ) << 1;
		auto Z = (topology_type::_DK) << 1;

		coordinates_type r = std::get<1>(idx);
		compact_index_type s = std::get<0>(idx);

		return

		get_value(f, ((s + X) + Y) + Z) * (r[0]) * (r[1]) * (r[2]) //
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
	static inline auto Gather_(mesh_type const & mesh, std::integral_constant<unsigned int ,VERTEX>, TF const &f, coordinates_type r)
	DECL_RET_TYPE(Gather_impl_(f, mesh.CoordinatesGlobalToLocal(r, 0UL) ))

	template<typename TF>
	static inline auto Gather_(mesh_type const& mesh, std::integral_constant<unsigned int ,EDGE>, TF const &f, coordinates_type r)
	DECL_RET_TYPE(
			make_ntuple(

					Gather_impl_(f, mesh.CoordinatesGlobalToLocal(r, (topology_type::_DI)) ),
					Gather_impl_(f, mesh.CoordinatesGlobalToLocal(r, (topology_type::_DJ)) ),
					Gather_impl_(f, mesh.CoordinatesGlobalToLocal(r, (topology_type::_DK)) )
			))

	template<typename TF>
	static inline auto Gather_(mesh_type const& mesh, std::integral_constant<unsigned int ,FACE>, TF const &f, coordinates_type r)
	DECL_RET_TYPE( make_ntuple(

					Gather_impl_(f, mesh.CoordinatesGlobalToLocal(r,((topology_type::_DJ | topology_type::_DK))) ),
					Gather_impl_(f, mesh.CoordinatesGlobalToLocal(r,((topology_type::_DK | topology_type::_DI))) ),
					Gather_impl_(f, mesh.CoordinatesGlobalToLocal(r,((topology_type::_DI | topology_type::_DJ))) )
			) )

	template<typename TF>
	static inline auto Gather_(mesh_type const& mesh, std::integral_constant<unsigned int ,VOLUME>, TF const &f, coordinates_type const & x)
	DECL_RET_TYPE(Gather_impl_(f, mesh.CoordinatesGlobalToLocal(x, (topology_type::_DA)) ))
private:
	template<typename TF, typename IDX, typename TV>
	static inline void Scatter_impl_(TF *f, IDX const& idx, TV & v)
	{

		auto X = (topology_type::_DI) << 1;
		auto Y = (topology_type::_DJ) << 1;
		auto Z = (topology_type::_DK) << 1;

		coordinates_type r = std::get<1>(idx);
		compact_index_type s = std::get<0>(idx);

		get_value(*f, ((s + X) + Y) + Z) += v * (r[0]) * (r[1]) * (r[2]);
		get_value(*f, (s + X) + Y) += v * (r[0]) * (r[1]) * (1.0 - r[2]);
		get_value(*f, (s + X) + Z) += v * (r[0]) * (1.0 - r[1]) * (r[2]);
		get_value(*f, s + X) += v * (r[0]) * (1.0 - r[1]) * (1.0 - r[2]);
		get_value(*f, (s + Y) + Z) += v * (1.0 - r[0]) * (r[1]) * (r[2]);
		get_value(*f, s + Y) += v * (1.0 - r[0]) * (r[1]) * (1.0 - r[2]);
		get_value(*f, s + Z) += v * (1.0 - r[0]) * (1.0 - r[1]) * (r[2]);
		get_value(*f, s) += v * (1.0 - r[0]) * (1.0 - r[1]) * (1.0 - r[2]);
	}
public:
	template<typename TF, typename TV>
	static inline void Scatter_(mesh_type const& mesh, std::integral_constant<unsigned int ,VERTEX>, TF *f, coordinates_type const &x, TV const & v)
	{
		get_value(*f, std::get<0>(mesh.CoordinatesGlobalToLocalNGP(x, 0UL))) += v;
//		Scatter_impl_(f, mesh.CoordinatesGlobalToLocal(r, 0UL), v);
	}

	template<typename TF, typename TV>
	static inline void Scatter_(mesh_type const& mesh, std::integral_constant<unsigned int ,EDGE>, TF *f, coordinates_type const & x,
	        nTuple<3, TV> const & u)
	{
		Scatter_impl_(f, mesh.CoordinatesGlobalToLocal(x, (topology_type::_DI)), u[0]);
		Scatter_impl_(f, mesh.CoordinatesGlobalToLocal(x, (topology_type::_DJ)), u[1]);
		Scatter_impl_(f, mesh.CoordinatesGlobalToLocal(x, (topology_type::_DK)), u[2]);

	}

	template<typename TF, typename TV>
	static inline void Scatter_(mesh_type const& mesh, std::integral_constant<unsigned int ,FACE>, TF *f, coordinates_type const & x,
	        nTuple<3, TV> const & u)
	{

		Scatter_impl_(f, mesh.CoordinatesGlobalToLocal(x, ((topology_type::_DJ | topology_type::_DK))), u[0]);
		Scatter_impl_(f, mesh.CoordinatesGlobalToLocal(x, ((topology_type::_DK | topology_type::_DI))), u[1]);
		Scatter_impl_(f, mesh.CoordinatesGlobalToLocal(x, ((topology_type::_DI | topology_type::_DJ))), u[2]);
	}

	template<typename TF, typename TV>
	static inline void Scatter_(mesh_type const& mesh, std::integral_constant<unsigned int ,VOLUME>, TF *f, coordinates_type const & x,
	        TV const & v)
	{
		get_value(*f, std::get<0>(mesh.CoordinatesGlobalToLocalNGP(x, topology_type::_DA))) += v;
//		Scatter_impl_(f, mesh.CoordinatesGlobalToLocal(x, topology_type::_DA), v);
	}

	/***
	 *  @param z (x,v) tangent bundle space coordinates,  base on "mesh_type::geometry_type"
	 *  @param weight scatter weight
	 */
	template<unsigned int IFORM, typename TContainer, typename TZ, typename TW>
	static inline void Scatter(Field<mesh_type, IFORM, TContainer> *f, TZ const & Z, TW weight = 1.0)
	{
		Scatter_(f->mesh, std::integral_constant<unsigned int ,IFORM>(), f, std::get<0>(Z), std::get<1>(Z) * weight);
	}

	/***
	 *  @param x configure space coordiantes,   base on "mesh_type::geometry_type"
	 *  @return  field value(vector/scalar) on Cartesian configure space
	 */
	template<unsigned int IFORM, typename TContainer, typename ... Args>
	static inline auto Gather(Field<mesh_type, IFORM, TContainer> const &f, coordinates_type const & x)
	DECL_RET_TYPE ( Gather_(f.mesh, std::integral_constant<unsigned int ,IFORM>(), f, x))

	/***
	 *  @param z (x,v) tangent bundle space coordinates on  Cartesian configure space
	 *  @param weight scatter weight
	 */
	template<unsigned int IFORM, typename TContainer, typename TZ, typename TW>
	static inline void ScatterCartesian(Field<mesh_type, IFORM, TContainer> *f, TZ const&z, TW weight)
	{

		TZ Z = f->mesh.PushForward(z);
		Scatter_(f->mesh, std::integral_constant<unsigned int ,IFORM>(), f, std::get<0>(Z), std::get<1>(Z) * weight);
	}

	/***
	 *  @param x tangent bundle space coordinates on  Cartesian configure space
	 *  @return  field value(vector/scalar) on Cartesian configure space
	 */
//	template<unsigned int IFORM, typename TContainer>
//	static inline auto GatherCartesian(Field<mesh_type, IFORM, TContainer> const &f, coordinates_type const& x)
//	DECL_RET_TYPE (
//			std::get<1>(f.mesh.PullBack(std::make_tuple(f.mesh.MapTo(x),
//									Gather_(f.mesh, std::integral_constant<unsigned int ,IFORM>(), f, f.mesh.MapTo(x))))))
	/***
	 *  @param x tangent bundle space coordinates on  Cartesian configure space
	 *  @return  field value(vector/scalar) on Cartesian configure space
	 */
	template<unsigned int IFORM, typename TContainer>
	static inline typename Field<mesh_type, IFORM, TContainer>::field_value_type GatherCartesian(
	        Field<mesh_type, IFORM, TContainer> const &f, coordinates_type const& x)
	{
		auto y = f.mesh.MapTo(x);

		return std::move(std::get<1>(f.mesh.PullBack(std::make_tuple(std::move(y), Gather_(f.mesh,

		std::integral_constant<unsigned int, IFORM>(), f, y)))));
	}
}
;

}  // namespace simpla

#endif /* INTERPOLATOR_H_ */
