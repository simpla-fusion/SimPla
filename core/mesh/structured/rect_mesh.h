/**
 * @file  rect_mesh.h
 *
 *  created on: 2014-2-21
 *      Author: salmon
 */

#ifndef MESH_RECT_MESH_H_
#define MESH_RECT_MESH_H_

#include <stddef.h>
#include <algorithm>
#include <functional>
#include <iterator>
#include <limits>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>

#include "../../utilities/utilities.h"
#include "../../gtl/ntuple.h"
#include "../../gtl/primitives.h"
#include "../../gtl/containers/sp_hash_container.h"

#include "../mesh_ids.h"

namespace simpla
{

/**
 * @ingroup topology
 *  @brief  structured mesh, n-dimensional array
 *
 *## Cell Shape
 * - Voxel(hexahedron):
 * - Define
 *  \verbatim
 *                ^y
 *               /
 *        z     /
 *        ^    /
 *        |  110(6)----------111(7)
 *        |  /|              /|
 *        | / |             / |
 *        |/  |            /  |
 *       100(4)----------101(5)
 *        |   |           |   |
 *        |  010(2)-----------|--011(3)
 *        |  /            |  /
 *        | /             | /
 *        |/              |/
 *       000(0)----------001(1)---> x
 * \endverbatim
 *  - the unit cell width is 1;
 */
template<typename TCoordSystem, size_t ... NFLAGS>
struct RectMesh: public Domain<MeshIDs_<NFLAGS...>>,
		public std::enable_shared_from_this<RectMesh<TCoordSystem, NFLAGS...>>
{

	typedef TCoordSystem coordinates_system;

	typedef RectMesh<TCoordSystem, NFLAGS...> this_type;

	typedef MeshIDs_<NFLAGS...> topology_type;

	typedef Domain<topology_type> domain_type;

	using topology_type::ndims;

	using typename topology_type::index_type;

	using typename topology_type::index_tuple;

	using typename topology_type::id_type;

	using typename topology_type::coordinates_type;

private:
	bool is_valid_ = false;

	coordinates_type m_coord_orig_ /*= { 0, 0, 0 }*/;

	coordinates_type m_coords_min_ = { 0, 0, 0 };

	coordinates_type m_coords_max_ = { 1, 1, 1 };

	coordinates_type m_dx_ /*= { 0, 0, 0 }*/;

	coordinates_type m_to_topology_scale_;

	coordinates_type m_from_topology_scale_;

public:

//***************************************************************************************************

	RectMesh()
	{
	}

	~RectMesh()
	{
	}

	RectMesh(this_type const & other) :
			domain_type(other)
	{
	}

	void swap(this_type & other)
	{
		domain_type::swap(other);
	}
	this_type & operator=(const this_type& other)
	{
		this_type(other).swap(*this);
		return *this;
	}

	template<typename OS>
	OS & print(OS &os) const
	{
		os

		<< " Type = \"" << get_type_as_string() << "\", " << std::endl

		<< " Min = " << m_coords_min_ << " ," << std::endl

		<< " Max  = " << m_coords_max_ << "," << std::endl

		<< " dx  = " << m_dx_ << "," << std::endl

		<< " dt  = " << m_dt_ << "," << std::endl;

		domain_type::print(os);

		return os;

	}

	static std::string get_type_as_string()
	{
		return "RectMesh<" + coordinates_system::get_type_as_string() + ">";
	}

	constexpr bool is_valid() const
	{
		return is_valid_;
	}

	template<typename T0, typename T1>
	void extents(T0 const& pmin, T1 const& pmax)
	{
		m_coords_min_ = pmin;
		m_coords_max_ = pmax;
	}

	inline auto extents() const
	DECL_RET_TYPE (std::make_pair(m_coords_min_, m_coords_max_))

	coordinates_type const & dx() const
	{
		return m_dx_;
	}

	void deploy();
	/**
	 * 	@name  Time
	 *  @{
	 *
	 */

private:
	Real m_time_ = 0;
	Real m_dt_ = 1.0;
	Real m_CFL_ = 0.5;

public:
	void next_timestep()
	{
		m_time_ += m_dt_;
	}

	// Time

	Real dt() const
	{
		return m_dt_;
	}
	void dt(Real pdt)
	{
		m_dt_ = pdt;
	}

	void time(Real p_time)
	{
		m_time_ = p_time;
	}
	Real time() const
	{
		return m_time_;
	}

	/** @} */

	/** @name Volume
	 * @{
	 */
	Real m_volume_[9];
	Real m_inv_volume_[9];
	Real m_dual_volume_[9];
	Real m_inv_dual_volume_[9];

	void deploy_volume();

	constexpr Real volume(id_type s) const
	{
		return m_volume_[topology_type::node_id(s)]
				* coordinates_system::volume_factor(coordinates(s),
						topology_type::node_id(s));
	}

	constexpr Real dual_volume(id_type s) const
	{
		return m_dual_volume_[topology_type::node_id(s)]
				* coordinates_system::dual_volume_factor(coordinates(s),
						topology_type::node_id(s));;
	}

	constexpr Real cell_volume(id_type s) const
	{
		return volume(s | MeshIDs::_DA);
	}

	constexpr Real inv_volume(id_type s) const
	{
		return m_inv_volume_[topology_type::node_id(s)]
				* coordinates_system::inv_volume_factor(coordinates(s),
						topology_type::node_id(s));
	}

	constexpr Real inv_dual_volume(id_type s) const
	{
		return m_inv_dual_volume_[topology_type::node_id(s)]
				* coordinates_system::inv_dual_volume_factor(coordinates(s),
						topology_type::node_id(s));
	}
	/**@}*/

	/**
	 * @name  Coordinates map
	 * @{
	 **/
	coordinates_type coordinates(id_type const & s) const
	{
		return std::move(
				coordinates_from_topology(topology_type::id_to_coordinates(s)));
	}

	template<size_t IFORM = 0>
	inline id_type coordinates_to_id(coordinates_type const &x, int n = 0) const
	{
		return topology_type::coordinates_to_id<IFORM>(
				coordinates_to_topology(x), n);
	}

	coordinates_type coordinates_from_topology(coordinates_type const &y) const
	{

		return coordinates_type(
				{

				std::fma(y[0] - topology_type::COORD_ZERO,
						m_from_topology_scale_[0], m_coord_orig_[0]),

				std::fma(y[1] - topology_type::COORD_ZERO,
						m_from_topology_scale_[1], m_coord_orig_[1]),

				std::fma(y[2] - topology_type::COORD_ZERO,
						m_from_topology_scale_[2], m_coord_orig_[2])

				});

	}
	coordinates_type coordinates_to_topology(coordinates_type const &x) const
	{
		return coordinates_type(
				{

				std::fma(x[0] - m_coord_orig_[0], m_to_topology_scale_[0],
						topology_type::COORD_ZERO),

				std::fma(x[1] - m_coord_orig_[1], m_to_topology_scale_[1],
						topology_type::COORD_ZERO),

				std::fma(x[2] - m_coord_orig_[2], m_to_topology_scale_[2],
						topology_type::COORD_ZERO)

				});

	}

	/**
	 * @bug: truncation error of coordinates transform larger than 1000
	 *     epsilon (1e4 epsilon for cylindrical coordinates)
	 * @param args
	 * @return
	 */
	template<typename ... Args>
	inline coordinates_type coordinates_local_to_global(Args && ... args) const
	{
		return std::move(
				coordinates_from_topology(
						topology_type::coordinates_local_to_global(
								std::forward<Args >(args)...)));
	}

	template<size_t IFORM>
	std::tuple<id_type, coordinates_type> coordinates_global_to_local(
			coordinates_type x, int n = 0) const
	{
		return std::move(
				topology_type::template coordinates_global_to_local<IFORM>(
						std::move(coordinates_to_topology(x)), n));
	}

	/** @} */

}
;

template<typename TCoord, size_t ... N>
void RectMesh<TCoord, N...>::deploy()
{
	index_tuple dims = domain_type::dimensions();

	for (size_t i = 0; i < ndims; ++i)
	{
		if (dims[i] > 0 && (m_coords_max_[i] - m_coords_min_[i]) > EPSILON)
		{

			m_dx_[i] = (m_coords_max_[i] - m_coords_min_[i])
					/ static_cast<Real>(dims[i]);

			m_to_topology_scale_ = dims[i]
					/ (m_coords_max_[i] - m_coords_min_[i]);

			m_from_topology_scale_[i] = (m_coords_max_[i] - m_coords_min_[i])
					/ dims[i];
		}
#ifdef  ENABLE_COMPLEX_COORDINATE_SYSTEM
		else if ((m_coords_max_[i] - m_coords_min_[i]) > EPSILON)
		{
			dims[i] = 1;
			m_dx_[i] = 0;
			m_to_topology_scale_ = 1.0 / (m_coords_max_[i] - m_coords_min_[i]);
			m_from_topology_scale_[i] = (m_coords_max_[i] - m_coords_min_[i])
			/ 1.0;
		}
#endif
		else
		{
			dims[i] = 1;

			m_dx_[i] = 0;

			m_coords_max_[i] = m_coords_min_[i];

			m_to_topology_scale_[i] = 0;
			m_from_topology_scale_[i] = 0;
		}

	}

	m_coord_orig_ = (m_coords_max_ + m_coords_min_) * 0.5;

	DEFINE_PHYSICAL_CONST

	Real safe_dt = m_CFL_
			* std::sqrt(
					m_dx_[0] * m_dx_[0] + m_dx_[1] * m_dx_[1]
							+ m_dx_[2] * m_dx_[2]) / speed_of_light;

	if (m_dt_ > safe_dt)
	{
		WARNING << ("  Courant–Friedrichs–Lewy (CFL) !") << std::endl;
	}

	deploy_volume();

	domain_type::dimensions(&dims[0]);

	domain_type::deploy();

	is_valid_ = true;

	VERBOSE << get_type_as_string() << " is deployed!" << std::endl;

}

template<typename TCoord, size_t ... N>
void RectMesh<TCoord, N...>::deploy_volume()
{

	for (size_t i = 0; i < ndims; ++i)
	{

		if (m_dx_[i] <= EPSILON)
		{

			m_volume_[1UL << (ndims - i - 1)] = 1.0;

			m_dual_volume_[7 - (1UL << (ndims - i - 1))] = 1.0;

			m_inv_volume_[1UL << (ndims - i - 1)] = 1.0;

			m_inv_dual_volume_[7 - (1UL << (ndims - i - 1))] = 1.0;

		}
		else
		{

			m_volume_[1UL << (ndims - i - 1)] = m_dx_[i];

			m_dual_volume_[7 - (1UL << (ndims - i - 1))] = m_dx_[i];

			m_inv_volume_[1UL << (ndims - i - 1)] = 1.0 / m_dx_[i];

			m_inv_dual_volume_[7 - (1UL << (ndims - i - 1))] = 1.0 / m_dx_[i];

		}
	}

//	/**
//	 *\verbatim
//	 *                ^y
//	 *               /
//	 *        z     /
//	 *        ^    /
//	 *        |  110-------------111
//	 *        |  /|              /|
//	 *        | / |             / |
//	 *        |/  |            /  |
//	 *       100--|----------101  |
//	 *        | m |           |   |
//	 *        |  010----------|--011
//	 *        |  /            |  /
//	 *        | /             | /
//	 *        |/              |/
//	 *       000-------------001---> x
//	 *
//	 *\endverbatim
//	 */
//
//	m_volume_[1] /* 001 */= m_dx_[0];
//	m_volume_[2] /* 010 */= m_dx_[1];
//	m_volume_[4] /* 100 */= m_dx_[2];

	m_volume_[3] /* 011 */= m_volume_[1] * m_volume_[2];
	m_volume_[5] /* 101 */= m_volume_[4] * m_volume_[1];
	m_volume_[6] /* 110 */= m_volume_[2] * m_volume_[4];

	m_volume_[7] /* 111 */= m_volume_[1] * m_volume_[2] * m_volume_[4];

//	m_dual_volume_[6] /* 001 */= m_dx_[0];
//	m_dual_volume_[5] /* 010 */= m_dx_[1];
//	m_dual_volume_[3] /* 100 */= m_dx_[2];

	m_dual_volume_[4] /* 011 */= m_dual_volume_[6] * m_dual_volume_[5];
	m_dual_volume_[2] /* 101 */= m_dual_volume_[3] * m_dual_volume_[6];
	m_dual_volume_[1] /* 110 */= m_dual_volume_[5] * m_dual_volume_[3];

	m_dual_volume_[0] /* 111 */= m_dual_volume_[6] * m_dual_volume_[5]
			* m_dual_volume_[3];

//	m_inv_volume_[1] /* 001 */= m_inv_dx_[0];
//	m_inv_volume_[2] /* 010 */= m_inv_dx_[1];
//	m_inv_volume_[4] /* 100 */= m_inv_dx_[2];

	m_inv_volume_[3] /* 011 */= m_inv_volume_[1] * m_inv_volume_[2];
	m_inv_volume_[5] /* 101 */= m_inv_volume_[4] * m_inv_volume_[1];
	m_inv_volume_[6] /* 110 */= m_inv_volume_[2] * m_inv_volume_[4];
	m_inv_volume_[7] /* 111 */= m_inv_volume_[1] * m_inv_volume_[2]
			* m_inv_volume_[4];

//	m_inv_dual_volume_[6] /* 001 */= m_inv_dx_[0];
//	m_inv_dual_volume_[5] /* 010 */= m_inv_dx_[1];
//	m_inv_dual_volume_[3] /* 100 */= m_inv_dx_[2];

	m_inv_dual_volume_[4] /* 011 */= m_inv_dual_volume_[6]
			* m_inv_dual_volume_[5];
	m_inv_dual_volume_[2] /* 101 */= m_inv_dual_volume_[3]
			* m_inv_dual_volume_[6];
	m_inv_dual_volume_[1] /* 110 */= m_inv_dual_volume_[5]
			* m_inv_dual_volume_[3];

	m_inv_dual_volume_[0] /* 111 */= m_inv_dual_volume_[6]
			* m_inv_dual_volume_[5] * m_inv_dual_volume_[3];
}
}
// namespace simpla

//}  // namespace std

#endif /* MESH_RECT_MESH_H_ */
