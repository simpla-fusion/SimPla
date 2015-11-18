/*
 * @file structured.h
 *
 *  Created on: 2015-3-10
 *      Author: salmon
 */

#ifndef CORE_MESH_STRUCTURED_H_
#define CORE_MESH_STRUCTURED_H_

#include <stddef.h>
#include <algorithm>
#include <cmath>
#include <cstdbool>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "coordinate_system_geo.h"
#include "../../physics/physical_constants.h"
#include "../../gtl/utilities/log.h"
#include "manifold_traits.h"
#include "../mesh_ids.h"
#include "../policy.h"
#include "structured.h"
//#include "structured/topology.h"
//#include "structured/select.h"

namespace simpla
{

namespace tags
{
struct structured;

}

template<typename ...> struct Mesh;

/**
 * @ingroup geometry
 *  @brief  structured geometry, n-dimensional array
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
template<typename CoordinateSystem>
struct Mesh<CoordinateSystem, tags::RectMesh> :
		public Topology<tags::RectMesh>,
		public std::enable_shared_from_this<
				Mesh<CoordinateSystem, simpla::tags::RectMesh> >
{
	typedef CoordinateSystem cs_type;

	typedef Topology<tags::RectMesh> topology_type;

	typedef Mesh<cs_type, tags::RectMesh> this_type;

	static constexpr size_t ndims = geometry::traits::dimension<cs_type>::value;
	static constexpr size_t ZAXIS = geometry::traits::ZAxis<cs_type>::value;

	using typename topology_type::index_type;
	using typename topology_type::index_tuple;
	using typename topology_type::range_type;

	typedef traits::point_type_t<topology_type> topology_point_type;

	typedef traits::id_type_t<topology_type> id_type;

	typedef geometry::traits::scalar_type_t<cs_type> scalar_type;

	typedef geometry::traits::point_type_t<cs_type> point_type;

	typedef geometry::traits::vector_type_t<cs_type> vector_type;


	using topology_type::dimensions;

private:

	geometry::mertic<cs_type> m_metric_;

	bool m_is_valid_ = false;

	topology_point_type m_from_topology_orig_ /*= { 0, 0, 0 }*/;

	topology_point_type m_to_topology_orig_ /*= { 0, 0, 0 }*/;

	point_type m_coords_min_ = {0, 0, 0};

	point_type m_coords_max_ = {1, 1, 1};

	vector_type m_dx_; //!< width of cell, but m_dx_[i]=0 when m_dims_[i]==1

	vector_type m_delta_; //!< equiv. m_dx_, but m_delta_[i]=1 when m_dims_[i]==1

	topology_point_type m_to_topology_scale_;

	topology_point_type m_from_topology_scale_;


public:

//***************************************************************************************************

	Mesh()
	{
	}

	~Mesh()
	{
	}

	Mesh(this_type const &other) : topology_type(other)
	{
	}

	void swap(this_type &other)
	{
	}

	this_type &operator=(const this_type &other)
	{
		this_type(other).swap(*this);
		return *this;
	}

	template<typename TDict>
	void load(TDict const &dict)
	{
		topology_type::dimensions(dict["Dimensions"].as(index_tuple({10, 10, 10})));

		extents(dict["Box"].template as<std::tuple<point_type, point_type> >());

		dt(dict["dt"].template as<Real>(1.0));
	}

	template<typename OS>
	OS &print(OS &os) const
	{
		DEFINE_PHYSICAL_CONST

		Real safe_dt = 0.5
				* std::sqrt(
				(m_dx_[0] * m_dx_[0] + m_dx_[1] * m_dx_[1]
						+ m_dx_[2] * m_dx_[2])) / speed_of_light;

		os

				<< " Type = \"" << get_type_as_string() << "\", " << std::endl

				<< " Min \t= " << m_coords_min_ << " ," << std::endl

				<< " Max \t= " << m_coords_max_ << "," << std::endl

				<< " dx  \t= " << m_dx_ << ", " << std::endl

				<< " dt \t= " << m_dt_ << "," << std::endl

				<< "-- [ Courant–Friedrichs–Lewy (CFL)  Suggested value: " << safe_dt
				<< "]" << std::endl

				<< " Dimensions\t= " << dimensions() << "," << std::endl;

		return os;

	}

	point_type epsilon() const
	{
		return topology_type::EPSILON * m_from_topology_scale_;
	}


	static std::string get_type_as_string()
	{
		return "Mesh<" + traits::type_id<cs_type>::name() + ">";
	}

	constexpr bool is_valid() const
	{
		return m_is_valid_;
	}

	template<typename T0, typename T1>
	void extents(T0 const &pmin, T1 const &pmax)
	{
		m_coords_min_ = pmin;
		m_coords_max_ = pmax;
	}

	template<typename T0>
	void extents(T0 const &box)
	{
		extents(traits::get<0>(box), traits::get<1>(box));
	}

	constexpr std::pair<point_type, point_type> extents() const
	{
		return (std::make_pair(m_coords_min_, m_coords_max_));
	}

	constexpr std::pair<point_type, point_type> local_extents() const
	{
		return (std::make_pair(point(topology_type::m_id_local_min_), point(topology_type::m_id_local_max_)));
	}

	vector_type const &dx() const
	{
		return m_dx_;
	}


	void deploy(size_t const *gw = nullptr);

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
	void next_time_step()
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

	Real volume_(id_type s) const
	{
		return m_metric_.dual_volume(topology_type::node_id(s), point(s),
				m_delta_);
	}

	Real dual_volume_(id_type s) const
	{

		return m_metric_.dual_volume(topology_type::node_id(s), point(s),
				m_delta_);
	}

	Real volume(id_type s) const
	{
		return m_volume_[topology_type::node_id(s)] * volume_(s);
	}

	Real dual_volume(id_type s) const
	{
		return m_dual_volume_[topology_type::node_id(s)] * dual_volume_(s);
	}

	Real inv_volume(id_type s) const
	{
		return m_inv_volume_[topology_type::node_id(s)] / volume_(s);
	}

	Real inv_dual_volume(id_type s) const
	{
		return m_inv_dual_volume_[topology_type::node_id(s)] / dual_volume_(s);
	}

	/**@}*/

	/**
	 * @name  Coordinate map
	 * @{
	 *
	 *        Topology Manifold       BaseManifold Manifold
	 *                        map
	 *              M      ---------->      G
	 *              x                       y
	 **/
	point_type point(id_type const &s) const
	{
		return std::move(map(topology_type::point(s)));
	}

	point_type map(topology_point_type const &x) const
	{

		return point_type(
				{

						std::fma(traits::get<0>(x), m_from_topology_scale_[0],
								m_from_topology_orig_[0]),

						std::fma(traits::get<1>(x), m_from_topology_scale_[1],
								m_from_topology_orig_[1]),

						std::fma(traits::get<2>(x), m_from_topology_scale_[2],
								m_from_topology_orig_[2])

				});

	}

	topology_point_type inv_map(point_type const &y) const
	{

		return topology_point_type(
				{

						std::fma(traits::get<0>(y), m_to_topology_scale_[0],
								m_to_topology_orig_[0]),

						std::fma(traits::get<1>(y), m_to_topology_scale_[1],
								m_to_topology_orig_[1]),

						std::fma(traits::get<2>(y), m_to_topology_scale_[2],
								m_to_topology_orig_[2])

				});
	}

	template<typename TFun>
	auto pull_back(point_type const &x, TFun const &fun) const
	DECL_RET_TYPE((fun(map(x))))

	template<typename TFun>
	auto push_forward(point_type const &y, TFun const &fun) const
	DECL_RET_TYPE((fun(inv_map(y))))

	Vec3 pull_back(point_type const &y, vector_type const &u) const
	{
		return inv_map(y + u) - inv_map(y);
	}

	Vec3 push_forward(point_type const &x, vector_type const &v) const
	{
		return map(x + v) - map(x);
	}

	template<typename TX>
	topology_point_type coordinates_to_topology(TX const &y) const
	{
		return inv_map(y);
	}

	template<typename TX>
	point_type coordinates_from_topology(TX const &x) const
	{
		return map(x);
	}

	/**
	 * @bug: truncation error of coordinates transform larger than 1000
	 *     epsilon (1e4 epsilon for cylindrical coordinates)
	 * @param args
	 * @return
	 */
	point_type coordinates_local_to_global(
			std::tuple<id_type, topology_point_type> const &t) const
	{
		return std::move(map(topology_type::coordinates_local_to_global(t)));
	}

	std::tuple<id_type, topology_point_type> coordinates_global_to_local(
			point_type x, int n_id = 0) const
	{
		return std::move(
				topology_type::coordinates_global_to_local(
						coordinates_to_topology(x), n_id));
	}

	/** @} */


};

template<typename CoordinateSystem>
void Mesh<CoordinateSystem, tags::RectMesh>::deploy(size_t const *gw)
{
	nTuple<id_type, ndims> dims = topology_type::dimensions();

	for (int i = 0; i < ndims; ++i)
	{
		if (dims[i] > 1 && (m_coords_max_[i] - m_coords_min_[i]) > EPSILON)
		{

			m_dx_[i] = (m_coords_max_[i] - m_coords_min_[i])
					/ static_cast<Real>(dims[i]);

			m_delta_[i] = m_dx_[i]; // this is the correct one

			m_to_topology_scale_[i] = static_cast<Real>(dims[i])
					/ (m_coords_max_[i] - m_coords_min_[i])
					* topology_type::COORDINATES_MESH_FACTOR;

			m_from_topology_scale_[i] = (m_coords_max_[i] - m_coords_min_[i])
					/ static_cast<Real>(dims[i])
					/ topology_type::COORDINATES_MESH_FACTOR;

			m_to_topology_orig_[i] = -m_coords_min_[i] * m_to_topology_scale_[i]
					+ topology_type::INDEX_ZERO;

			m_from_topology_orig_[i] = m_coords_min_[i]
					- m_from_topology_scale_[i] * topology_type::INDEX_ZERO;

		}
//#ifdef  ENABLE_COMPLEX_COORDINATE_SYSTEM
//		else if ((m_coords_max_[i] - m_coords_min_[i]) > EPSILON)
//		{
//			m_index_dimensions_[i] = 1;
//			m_dx_[i] = 0;
//			m_to_topology_scale_ = 1.0 / (m_coords_max_[i] - m_coords_min_[i]);
//			m_from_topology_scale_[i] = (m_coords_max_[i] - m_coords_min_[i])
//			/ 1.0;
//		}
//#endif
		else
		{
			dims[i] = 1;

			m_dx_[i] = 0;

			m_delta_[i] = 1.0;

			m_coords_max_[i] = m_coords_min_[i];

			m_to_topology_scale_[i] = 0;

			m_from_topology_scale_[i] = 0;

			m_to_topology_orig_[i] = -m_coords_min_[i] * m_to_topology_scale_[i]
					+ topology_type::INDEX_ZERO;

			m_from_topology_orig_[i] = -m_from_topology_scale_[i]
					* topology_type::INDEX_ZERO + m_coords_min_[i];
		}

	}

	topology_type::dimensions(dims);
	topology_type::deploy(gw);

	/**
	 *  deploy volume
	 **/

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
	m_volume_[0] = 1.0;

	m_volume_[1/* 001*/] = (m_dx_[0] <= EPSILON) ? 1 : m_dx_[0];
	m_volume_[2/* 010*/] = (m_dx_[1] <= EPSILON) ? 1 : m_dx_[1];
	m_volume_[4/* 100*/] = (m_dx_[2] <= EPSILON) ? 1 : m_dx_[2];

	m_volume_[3] /* 011 */= m_volume_[1] * m_volume_[2];
	m_volume_[5] /* 101 */= m_volume_[4] * m_volume_[1];
	m_volume_[6] /* 110 */= m_volume_[2] * m_volume_[4];

	m_volume_[7] /* 111 */= m_volume_[1] * m_volume_[2] * m_volume_[4];

	m_dual_volume_[7] = 1.0;

	m_dual_volume_[6] = (m_dx_[0] <= EPSILON) ? 1 : m_dx_[0];
	m_dual_volume_[5] = (m_dx_[1] <= EPSILON) ? 1 : m_dx_[1];
	m_dual_volume_[3] = (m_dx_[2] <= EPSILON) ? 1 : m_dx_[2];

	m_dual_volume_[4] /* 011 */= m_dual_volume_[6] * m_dual_volume_[5];
	m_dual_volume_[2] /* 101 */= m_dual_volume_[3] * m_dual_volume_[6];
	m_dual_volume_[1] /* 110 */= m_dual_volume_[5] * m_dual_volume_[3];

	m_dual_volume_[0] /* 111 */= m_dual_volume_[6] * m_dual_volume_[5]
			* m_dual_volume_[3];

	m_inv_volume_[7] = 1.0;

	m_inv_volume_[1/* 001 */] = (m_dx_[0] <= EPSILON) ? 1 : 1.0 / m_dx_[0];
	m_inv_volume_[2/* 010 */] = (m_dx_[1] <= EPSILON) ? 1 : 1.0 / m_dx_[1];
	m_inv_volume_[4/* 100 */] = (m_dx_[2] <= EPSILON) ? 1 : 1.0 / m_dx_[2];

	m_inv_volume_[3] /* 011 */= m_inv_volume_[1] * m_inv_volume_[2];
	m_inv_volume_[5] /* 101 */= m_inv_volume_[4] * m_inv_volume_[1];
	m_inv_volume_[6] /* 110 */= m_inv_volume_[2] * m_inv_volume_[4];
	m_inv_volume_[7] /* 111 */= m_inv_volume_[1] * m_inv_volume_[2]
			* m_inv_volume_[4];

	m_inv_dual_volume_[7] = 1.0;

	m_inv_dual_volume_[6/* 110 */] = (m_dx_[0] <= EPSILON) ? 1 : 1.0 / m_dx_[0];
	m_inv_dual_volume_[5/* 101 */] = (m_dx_[1] <= EPSILON) ? 1 : 1.0 / m_dx_[1];
	m_inv_dual_volume_[3/* 001 */] = (m_dx_[2] <= EPSILON) ? 1 : 1.0 / m_dx_[2];

	m_inv_dual_volume_[4] /* 011 */= m_inv_dual_volume_[6]
			* m_inv_dual_volume_[5];
	m_inv_dual_volume_[2] /* 101 */= m_inv_dual_volume_[3]
			* m_inv_dual_volume_[6];
	m_inv_dual_volume_[1] /* 110 */= m_inv_dual_volume_[5]
			* m_inv_dual_volume_[3];

	m_inv_dual_volume_[0] /* 111 */= m_inv_dual_volume_[6]
			* m_inv_dual_volume_[5] * m_inv_dual_volume_[3];


	m_is_valid_ = true;

	VERBOSE << get_type_as_string() << " is deployed!" << std::endl;

}

}
// namespace simpla

#endif /* CORE_MESH_STRUCTURED_H_ */
