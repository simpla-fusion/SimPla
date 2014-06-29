/*
 * interpolator.h
 *
 *  Created on: 2014年4月17日
 *      Author: salmon
 */

#ifndef INTERPOLATOR_H_
#define INTERPOLATOR_H_

#include <cstddef>

#include "../utilities/primitives.h"

namespace simpla
{

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
	static inline typename TF::value_type Gather_impl_(TF const & f, TIDX idx, compact_index_type shift)
	{

		auto X = topology_type::DeltaIndex(0, shift);
		auto Y = topology_type::DeltaIndex(1, shift);
		auto Z = topology_type::DeltaIndex(2, shift);

		coordinates_type r = std::get<1>(idx);

		auto s = std::get<0>(idx) + topology_type::DeltaIndex(shift);

		return

		f[((s + X) + Y) + Z] * (r[0]) * (r[1]) * (r[2]) //
		+ f[((s + X) + Y) - Z] * (r[0]) * (r[1]) * (1.0 - r[2]) //
		+ f[((s + X) - Y) + Z] * (r[0]) * (1.0 - r[1]) * (r[2]) //
		+ f[((s + X) - Y) - Z] * (r[0]) * (1.0 - r[1]) * (1.0 - r[2]) //
		+ f[((s - X) + Y) + Z] * (1.0 - r[0]) * (r[1]) * (r[2]) //
		+ f[((s - X) + Y) - Z] * (1.0 - r[0]) * (r[1]) * (1.0 - r[2]) //
		+ f[((s - X) - Y) + Z] * (1.0 - r[0]) * (1.0 - r[1]) * (r[2]) //
		+ f[((s - X) - Y) - Z] * (1.0 - r[0]) * (1.0 - r[1]) * (1.0 - r[2]);
	}
public:
	template<typename TF>
	static inline auto Gather_(mesh_type const & mesh, Int2Type<VERTEX>, TF const &f, coordinates_type r)
	DECL_RET_TYPE(Gather_impl_(f, mesh.CoordinatesGlobalToLocal(r, 0UL), 0UL))

	template<typename TF>
	static inline auto Gather_(mesh_type const& mesh, Int2Type<EDGE>, TF const &f, coordinates_type r)
	DECL_RET_TYPE(
			make_ntuple(

					Gather_impl_(f, mesh.CoordinatesGlobalToLocal(r, (topology_type::_DI)), (topology_type::_DI)),
					Gather_impl_(f, mesh.CoordinatesGlobalToLocal(r, (topology_type::_DJ)), (topology_type::_DJ)),
					Gather_impl_(f, mesh.CoordinatesGlobalToLocal(r, (topology_type::_DK)), (topology_type::_DK))
			))

	template<typename TF>
	static inline auto Gather_(mesh_type const& mesh, Int2Type<FACE>, TF const &f, coordinates_type r)

	DECL_RET_TYPE( make_ntuple(

					Gather_impl_(f, mesh.CoordinatesGlobalToLocal(r,((topology_type::_DJ | topology_type::_DK))),
							((topology_type::_DJ | topology_type::_DK))),
					Gather_impl_(f, mesh.CoordinatesGlobalToLocal(r,((topology_type::_DK | topology_type::_DI))),
							((topology_type::_DK | topology_type::_DI))),
					Gather_impl_(f, mesh.CoordinatesGlobalToLocal(r,((topology_type::_DI | topology_type::_DJ))),
							((topology_type::_DI | topology_type::_DJ)))
			) )

	template<typename TF>
	static inline auto Gather_(Int2Type<VOLUME>, TF const &f, mesh_type const & mesh, coordinates_type r)
	DECL_RET_TYPE(Gather_impl_(f, mesh.CoordinatesGlobalToLocal(r, (topology_type::_DA)), (topology_type::_DA)))
private:
	template<typename TF, typename IDX, typename TV>
	static inline void Scatter_impl_(TF *f, IDX const& idx, TV & v, compact_index_type shift)
	{

		mesh_type const & mesh = f->mesh;
		auto X = topology_type::DeltaIndex(0, shift);
		auto Y = topology_type::DeltaIndex(1, shift);
		auto Z = topology_type::DeltaIndex(2, shift);

		coordinates_type r = std::get<1>(idx);
		auto s = std::get<0>(idx) + topology_type::DeltaIndex(shift);

		f[((s + X) + Y) + Z] += v * (r[0]) * (r[1]) * (r[2]);
		f[((s + X) + Y) - Z] += v * (r[0]) * (r[1]) * (1.0 - r[2]);
		f[((s + X) - Y) + Z] += v * (r[0]) * (1.0 - r[1]) * (r[2]);
		f[((s + X) - Y) - Z] += v * (r[0]) * (1.0 - r[1]) * (1.0 - r[2]);
		f[((s - X) + Y) + Z] += v * (1.0 - r[0]) * (r[1]) * (r[2]);
		f[((s - X) + Y) - Z] += v * (1.0 - r[0]) * (r[1]) * (1.0 - r[2]);
		f[((s - X) - Y) + Z] += v * (1.0 - r[0]) * (1.0 - r[1]) * (r[2]);
		f[((s - X) - Y) - Z] += v * (1.0 - r[0]) * (1.0 - r[1]) * (1.0 - r[2]);
	}
public:
	template<typename TF, typename TV>
	static inline void Scatter_(mesh_type const& mesh, Int2Type<VERTEX>, TF *f, coordinates_type r, TV const & v)
	{
		Scatter_impl_(f, mesh.CoordinatesGlobalToLocal(r, 0UL), v, 0UL);
	}

	template<typename TF, typename TV>
	static inline void Scatter_(mesh_type const& mesh, Int2Type<EDGE>, TF *f, coordinates_type r,
			nTuple<3, TV> const & u)
	{
		Scatter_impl_(f, mesh.CoordinatesGlobalToLocal(r, (topology_type::_DK)), u[2], (topology_type::_DK));
		Scatter_impl_(f, mesh.CoordinatesGlobalToLocal(r, (topology_type::_DI)), u[0], (topology_type::_DI));
		Scatter_impl_(f, mesh.CoordinatesGlobalToLocal(r, (topology_type::_DJ)), u[1], (topology_type::_DJ));
	}

	template<typename TF, typename TV>
	static inline void Scatter_(mesh_type const& mesh, Int2Type<FACE>, TF *f, coordinates_type r,
			nTuple<3, TV> const & u)
	{

		Scatter_impl_(f, mesh.CoordinatesGlobalToLocal(r, ((topology_type::_DJ | topology_type::_DK))), u[2],
				(topology_type::_DJ | topology_type::_DK));
		Scatter_impl_(f, mesh.CoordinatesGlobalToLocal(r, ((topology_type::_DK | topology_type::_DI))), u[0],
				(topology_type::_DK | topology_type::_DI));
		Scatter_impl_(f, mesh.CoordinatesGlobalToLocal(r, ((topology_type::_DI | topology_type::_DJ))), u[1],
				(topology_type::_DI | topology_type::_DJ));
	}

	template<typename TF, typename TV>
	static inline void Scatter_(mesh_type const& mesh, Int2Type<VOLUME>, TF *f, coordinates_type r, TV const & v)
	{
		Scatter_impl_(f, mesh.CoordinatesGlobalToLocal(r, topology_type::_DA), v, topology_type::_DA);
	}

	/***
	 *  @param z (x,v) tangent bundle space coordinates on  Cartesian configure space
	 *  @param weight scatter weight
	 */
	template<int IFORM, typename TF, typename TZ, typename TS>
	static inline void Scatter(mesh_type const& mesh, Int2Type<IFORM>, TF *f, TZ const & z, scalar_type weight = 1.0)
	{
		auto Z = mesh.PushForward(z);
		Scatter_(mesh, Int2Type<IFORM>(), f, std::get<0>(Z), std::get<1>(Z) * weight);
	}

	/***
	 *  @param x tangent bundle space coordinates on  Cartesian configure space
	 *  @return  field value(vector/scalar) on Cartesian configure space
	 */
	template<int IFORM, typename TF>
	static inline auto Gather(mesh_type const & mesh, Int2Type<IFORM>, TF const &f, coordinates_type const & x)
	->decltype(Gather_(mesh,Int2Type<IFORM>(),f,x))
	{
		auto r = f.mesh.MapTo(x);
		return std::move(std::get<1>(f->mesh.PullBack(std::make_tuple(r, Gather_(Int2Type<IFORM>(), f, mesh, r)))));
	}

}
;

}  // namespace simpla

#endif /* INTERPOLATOR_H_ */
