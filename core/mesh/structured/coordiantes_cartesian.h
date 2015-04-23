/**
 * @file   cartesian.h
 *
 *  created on: 2014-3-13
 *      Author: salmon
 */

#ifndef CORE_MESH_STRUCTURED_COORDINATES_COORDIANTES_CARTESIAN_H_
#define CORE_MESH_STRUCTURED_COORDINATES_COORDIANTES_CARTESIAN_H_

#include <stddef.h>
#include <string>
#include <tuple>
#include <type_traits>
#include "../../utilities/utilities.h"
#include "../../physics/constants.h"
#include "../../physics/physical_constants.h"
#include "../../gtl/enable_create_from_this.h"
#include "../mesh.h"

namespace simpla
{
/**
 *  @ingroup geometry
 *
 *  \brief  Cartesian coordinates (X Y Z)
 *
 */
template<size_t NDIMS, size_t ZAXIS = CARTESIAN_ZAXIS>
struct CartesianCoordinates
{

public:

	typedef CartesianCoordinates<NDIMS, ZAXIS> this_type;

//	typedef MeshIDs_<NDIMS,ZAXIS> ids;
//
//	typedef typename ids::coordinates_type coordinates_type;
//	typedef typename ids::id_type id_type;

	static constexpr size_t ndims = NDIMS;

	static constexpr size_t XAxis = (ZAXIS + 1) % 3;
	static constexpr size_t YAxis = (ZAXIS + 2) % 3;
	static constexpr size_t ZAxis = ZAXIS;

	typedef Real scalar_type;

	typedef nTuple<Real, NDIMS> coordinates_type;

public:
	CartesianCoordinates()
	{
	}

	~CartesianCoordinates()
	{
	}

	static std::string get_type_as_string()
	{
		return "Cartesian Coordinates";
	}

	//***************************************************************************************************
	// Geometric properties
	// Metric
	//***************************************************************************************************

//	template<typename TDict, typename ...Others>
//	bool load(TDict const & dict, Others &&...others)
//	{
//		topology_type::load(dict,std::forward<Others>(others)...);
//
//		if (!topology_type::is_valid())
//		{
//			RUNTIME_ERROR("Topology is not initialized!");
//			return false;
//		}
//
//		if (!dict)
//		{
//			return false;
//		}
//
//		if (dict["Min"] && dict["Max"])
//		{
//
//			VERBOSE << "Load geometry : Cartesian " << std::endl;
//
//			extents(
//
//					dict["Min"].template as<nTuple<Real, ndims>>(),
//
//					dict["Max"].template as<nTuple<Real, ndims>>());
//
//			CFL_ = dict["CFL"].template as<Real>(0.5);
//
//			m_dt_ = dict["dt"].template as<Real>(1.0);
//
//		}
//		else
//		{
//			WARNING << "Configure Error: no Min or Max ";
//
//		}
//
//		return true;
//	}

//	inline coordinates_type dx(id_type s = 0UL) const
//	{
//		coordinates_type res;
//
//		auto d = topology_type::dimensions();
//
//		for (size_t i = 0; i < ndims; ++i)
//		{
//			res[i] = m_length_[i] / d[i];
//		}
//
//		return std::move(res);
//	}

	static nTuple<Real, 3> MapToCartesian(coordinates_type const &y)
	{
		nTuple<Real, 3> x;

		x[CARTESIAN_XAXIS] = y[XAxis];
		x[CARTESIAN_YAXIS] = y[YAxis];
		x[CARTESIAN_ZAXIS] = y[ZAxis];

		return std::move(x);
	}

	static coordinates_type MapFromCartesian(nTuple<Real, 3> const &x)
	{

		coordinates_type y;

		y[XAxis] = x[CARTESIAN_XAXIS];
		y[YAxis] = x[CARTESIAN_YAXIS];
		y[ZAxis] = x[CARTESIAN_ZAXIS];

		return std::move(y);
	}

	template<typename TV>
	static std::tuple<coordinates_type, TV> push_forward(
			std::tuple<coordinates_type, TV> const & Z)
	{
		return std::move(
				std::make_tuple(MapFromCartesian(std::get<0>(Z)),
						std::get<1>(Z)));
	}

	template<typename TV>
	static std::tuple<coordinates_type, TV> pull_back(
			std::tuple<coordinates_type, TV> const & R)
	{
		return std::move(
				std::make_tuple(MapToCartesian(std::get<0>(R)), std::get<1>(R)));
	}

	static coordinates_type Lie_trans(coordinates_type const & x,
			nTuple<Real, 3> const & v)
	{
		coordinates_type res;
		res = x + v;

		return std::move(res);
	}

	static coordinates_type Lie_trans(
			std::tuple<coordinates_type, nTuple<Real, 3> > const &Z,
			nTuple<Real, 3> const & v)
	{
		coordinates_type res;
		res = std::get<0>(Z) + v;

		return std::move(res);
	}
	/**
	 *
	 *   transform vector from Cartesian to Cartesian
	 *
	 * @param Z \f$\left(x,v\right)\f$\f$ u[XAixs] \partial_x +  u[YAixs] \partial_y + u[ZAixs] \partial_z \f$
	 * @param CartesianZAxis
	 * @return y u
	 *
	 */

	template<typename TV>
	static std::tuple<coordinates_type, nTuple<TV, ndims> > push_forward(
			std::tuple<coordinates_type, nTuple<TV, ndims> > const & Z)
	{
		coordinates_type y = MapFromCartesian(std::get<0>(Z));

		auto const & v = std::get<1>(Z);

		nTuple<TV, ndims> u;

		u[XAxis] = v[CARTESIAN_XAXIS];
		u[YAxis] = v[CARTESIAN_YAXIS];
		u[ZAxis] = v[CARTESIAN_ZAXIS];

		return (std::make_tuple(y, u));
	}

	/**
	 *
	 *  transform vector  from    Cylindrical to Cartesian
	 * @param R \f$\left(r, z ,\theta\right)\f$
	 *  \f$ u = u[RAixs] \partial_r +  u[1]  r[RAxis] \partial_theta + u[ZAixs] \partial_z\f$
	 * @param CartesianZAxis
	 * @return  x,\f$v = v[XAixs] \partial_x +  v[YAixs] \partial_y + v[ZAixs] \partial_z\f$
	 */
	template<typename TV>
	static std::tuple<coordinates_type, nTuple<TV, ndims> > pull_back(
			std::tuple<coordinates_type, nTuple<TV, ndims> > const & R,
			size_t CartesianZAxis = 2)
	{
		auto x = MapToCartesian(std::get<0>(R));
		auto const & u = std::get<1>(R);

		nTuple<TV, ndims> v;

		v[CARTESIAN_XAXIS] = u[XAxis];
		v[CARTESIAN_YAXIS] = u[YAxis];
		v[CARTESIAN_ZAXIS] = u[ZAxis];

		return std::move(std::make_tuple(x, v));
	}
	static constexpr Real inv_volume_factor(...)
	{
		return 1.0;
	}
	static constexpr Real volume_factor(...)
	{
		return 1.0;
	}
	static constexpr Real inv_dual_volume_factor(...)
	{
		return 1.0;
	}
	static constexpr Real dual_volume_factor(...)
	{
		return 1.0;
	}

//	template<size_t IFORM, typename TR>
//	auto select(TR range, coordinates_type const & xmin,
//			coordinates_type const & xmax) const
//			DECL_RET_TYPE((topology_type::template select<IFORM>(range,
//									this->coordinates_to_topology(xmin),
//									this->coordinates_to_topology(xmax))))
//
//	template<size_t IFORM, typename TR, typename ...Args>
//	auto select(TR range, Args && ...args) const
//	DECL_RET_TYPE((topology_type::template select<IFORM>(
//							range,std::forward<Args >(args)...)))
//
//	template<size_t IFORM>
//	auto select() const
//	DECL_RET_TYPE((this->topology_type:: template select<IFORM>()))

//***************************************************************************************************
// Volume
//***************************************************************************************************

	/**
	 *\verbatim
	 *                ^y
	 *               /
	 *        z     /
	 *        ^    /
	 *        |  110-------------111
	 *        |  /|              /|
	 *        | / |             / |
	 *        |/  |            /  |
	 *       100--|----------101  |
	 *        | m |           |   |
	 *        |  010----------|--011
	 *        |  /            |  /
	 *        | /             | /
	 *        |/              |/
	 *       000-------------001---> x
	 *
	 *\endverbatim
	 */

	scalar_type volume_[8] = { 1, // 000
			1, //001
			1, //010
			1, //011
			1, //100
			1, //101
			1, //110
			1 //111
			};
	scalar_type inv_volume_[8] = { 1, 1, 1, 1, 1, 1, 1, 1 };

	scalar_type dual_volume_[8] = { 1, 1, 1, 1, 1, 1, 1, 1 };

	scalar_type inv_dual_volume_[8] = { 1, 1, 1, 1, 1, 1, 1, 1 };

public:

	constexpr scalar_type cell_volume(coordinates_type const & x,
			coordinates_type const & dx) const
	{
		return dx[0] * dx[1] * dx[2];
	}
	constexpr scalar_type volume(coordinates_type const & x,
			coordinates_type const & dx) const
	{
		return 1.0;
	}
//	constexpr scalar_type inv_volume(id_type s) const
//	{
//		return topology_type::inv_volume(s)
//				* inv_volume_[topology_type::ele_suffix(s)];
//	}
//
//	constexpr scalar_type dual_volume(id_type s) const
//	{
//		return topology_type::dual_volume(s)
//				* dual_volume_[topology_type::ele_suffix(s)];
//	}
//	constexpr scalar_type inv_dual_volume(id_type s) const
//	{
//		return topology_type::inv_dual_volume(s)
//				* inv_dual_volume_[topology_type::ele_suffix(s)];
//	}
//
//	constexpr Real HodgeStarVolumeScale(id_type s) const
//	{
//		return 1.0;
//	}

}
;

}  // namespace simpla

#endif /* CORE_MESH_STRUCTURED_COORDINATES_COORDIANTES_CARTESIAN_H_ */
