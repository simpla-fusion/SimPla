/*
 * @file structured.h
 *
 *  Created on: 2015年3月10日
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

#include "../dataset/dataspace.h"
#include "../geometry/coordinate_system.h"

#include "../gtl/macro.h"
#include "../gtl/primitives.h"
#include "../gtl/type_traits.h"
#include "../parallel/mpi_comm.h"
#include "../parallel/mpi_update.h"
#include "../physics/physical_constants.h"
#include "../utilities/log.h"
#include "mesh_traits.h"
#include "mesh_ids.h"
#include "policy.h"

namespace simpla
{
namespace tags
{
struct structured;

}  // namespace tags

template<typename ... > struct Mesh;

/**
 * @ingroup mesh
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
template<typename CoordinateSystem>
struct Mesh<CoordinateSystem, simpla::tags::structured>

:	public MeshIDs_<geometry::traits::dimension<CoordinateSystem>::value>,
	public std::enable_shared_from_this<
			Mesh<CoordinateSystem, simpla::tags::structured> >
{
	typedef CoordinateSystem cs_type;

	typedef Mesh<cs_type, tags::structured> this_type;

	static constexpr size_t ndims = geometry::traits::dimension<cs_type>::value;
	static constexpr size_t ZAXIS = geometry::traits::ZAxis<cs_type>::value;

	typedef MeshIDs_<ndims> topology_type;
	using typename topology_type::index_type;
	using typename topology_type::index_tuple;
	using typename topology_type::range_type;

	typedef traits::point_type_t<topology_type> topology_point_type;

	typedef traits::id_type_t<topology_type> id_type;

	typedef geometry::traits::scalar_type_t<cs_type> scalar_type;

	typedef geometry::traits::point_type_t<cs_type> point_type;

	typedef geometry::traits::vector_type_t<cs_type> vector_type;

	typedef nTuple<id_type, ndims> id_tuple;

private:

	geometry::mertic<cs_type> m_metric_;

	static constexpr size_t DEFAULT_GHOST_WIDTH = 2;

	bool m_is_valid_ = false;

	bool m_is_distributed_ = false;

	topology_point_type m_from_topology_orig_ /*= { 0, 0, 0 }*/;

	topology_point_type m_to_toplogy_orig_ /*= { 0, 0, 0 }*/;

	point_type m_coords_min_ = { 0, 0, 0 };

	point_type m_coords_max_ = { 1, 1, 1 };

	vector_type m_dx_; //!< width of cell, but m_dx_[i]=0 when m_dims_[i]==1

	vector_type m_delta_; //!< equiv. m_dx_, but m_delta_[i]=1 when m_dims_[i]==1

	topology_point_type m_to_topology_scale_;

	topology_point_type m_from_topology_scale_;

	/**
	 *
	 *   -----------------------------5
	 *   |                            |
	 *   |     ---------------4       |
	 *   |     |              |       |
	 *   |     |  ********3   |       |
	 *   |     |  *       *   |       |
	 *   |     |  *       *   |       |
	 *   |     |  *       *   |       |
	 *   |     |  2********   |       |
	 *   |     1---------------       |
	 *   0-----------------------------
	 *
	 *	5-0 = dimensions
	 *	4-1 = e-d = ghosts
	 *	2-1 = counts
	 *
	 *	0 = id_begin
	 *	5 = id_end
	 *
	 *	1 = id_local_outer_begin
	 *	4 = id_local_outer_end
	 *
	 *	2 = id_local_inner_begin
	 *	3 = id_local_inner_end
	 *
	 *
	 */
//	id_type m_index_count_;
//
//	id_type m_index_dimensions_;
//
//	id_type m_index_offset_;
//
//	id_type m_index_local_dimensions_;
//
//	id_type m_index_local_offset_;
	id_type m_id_min_;

	id_type m_id_max_;

	id_type m_id_local_min_;

	id_type m_id_local_max_;

	id_type m_id_memory_min_;

	id_type m_id_memory_max_;

public:

//***************************************************************************************************

	Mesh()
	{
	}

	~Mesh()
	{
	}

	Mesh(this_type const & other)
			:

			m_id_min_(other.m_id_min_),

			m_id_max_(other.m_id_max_),

			m_id_local_min_(other.m_id_local_min_),

			m_id_local_max_(other.m_id_local_max_),

			m_id_memory_max_(other.m_id_memory_max_),

			m_id_memory_min_(other.m_id_memory_min_)

	{
	}

	void swap(this_type & other)
	{

		std::swap(m_id_min_, other.m_id_min_);
		std::swap(m_id_max_, other.m_id_max_);
		std::swap(m_id_local_min_, other.m_id_local_min_);
		std::swap(m_id_local_max_, other.m_id_local_max_);
		std::swap(m_id_memory_max_, other.m_id_memory_max_);
		std::swap(m_id_memory_min_, other.m_id_memory_min_);

	}
	this_type & operator=(const this_type& other)
	{
		this_type(other).swap(*this);
		return *this;
	}
	template<typename TDict>
	void load(TDict const & dict)
	{
		dimensions(dict["Dimensions"].as(index_tuple( { 10, 10, 10 })));

		extents(dict["Box"].template as<std::tuple<point_type, point_type> >());

		dt(dict["dt"].template as<Real>(1.0));
	}
	template<typename OS>
	OS & print(OS &os) const
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

				<< " Dimensions\t= " << dimensions() << "," << std::endl

				;

		return os;

	}

	point_type epsilon() const
	{
		return topology_type::EPSILON * m_from_topology_scale_;
	}

	std::tuple<id_tuple, id_tuple> index_box() const
	{
		return std::make_tuple(topology_type::unpack_index(m_id_min_),
				topology_type::unpack_index(m_id_max_));
	}
	std::tuple<id_tuple, id_tuple> local_index_box() const
	{
		return std::make_tuple(topology_type::unpack_index(m_id_local_min_),
				topology_type::unpack_index(m_id_local_max_));
	}

	range_type range(int nid = 0) const
	{

		return range_type(m_id_local_min_, m_id_local_max_, nid);
	}

	range_type range(point_type const & min, point_type const & max,
			int nid = 0) const
	{
//		geometry::model::Box<point_type> b;
//		bool success = geometry::intersection(
//				geometry::make_box(point(m_id_local_min_),
//						point(m_id_local_min_)), geometry::make_box(min, max),
//				b);
//		if (success)
//		{
//			return range_type(
//					traits::get<0>(
//							coordinates_global_to_local(traits::get<0>(b), nid)),
//					traits::get<1>(
//							coordinates_global_to_local(traits::get<1>(b), nid)),
//					nid);
//		}
//		else
//		{
//			return range_type();
//		}

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
	void extents(T0 const& pmin, T1 const& pmax)
	{
		m_coords_min_ = pmin;
		m_coords_max_ = pmax;
	}

	template<typename T0>
	void extents(T0 const& box)
	{
		extents(traits::get<0>(box), traits::get<1>(box));
	}

	constexpr std::pair<point_type, point_type> extents() const
	{
		return (std::make_pair(m_coords_min_, m_coords_max_));
	}

	constexpr std::pair<point_type, point_type> local_extents() const
	{
		return (std::make_pair(point(m_id_local_min_), point(m_id_local_max_)));
	}

	vector_type const & dx() const
	{
		return m_dx_;
	}

	template<typename TI> void dimensions(TI const & d)
	{

		m_id_min_ = topology_type::ID_ZERO;
		m_id_max_ = m_id_min_ + topology_type::pack_index(d);

	}
	index_tuple dimensions() const
	{
		return topology_type::unpack_index(m_id_max_ - m_id_min_);
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
	 *        Topology Manifold       Geometry Manifold
	 *                        map
	 *              M      ---------->      G
	 *              x                       y
	 **/
	point_type point(id_type const & s) const
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
						m_to_toplogy_orig_[0]),

				std::fma(traits::get<1>(y), m_to_topology_scale_[1],
						m_to_toplogy_orig_[1]),

				std::fma(traits::get<2>(y), m_to_topology_scale_[2],
						m_to_toplogy_orig_[2])

				});
	}

	template<typename TFun>
	auto pull_back(point_type const & x, TFun const & fun) const
	DECL_RET_TYPE((fun(map(x))))

	template<typename TFun>
	auto push_forward(point_type const & y, TFun const & fun) const
	DECL_RET_TYPE((fun(inv_map(y))))

	Vec3 pull_back(point_type const & y, vector_type const & u) const
	{
		return inv_map(y + u) - inv_map(y);
	}

	Vec3 push_forward(point_type const & x, vector_type const & v) const
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

	/**@name hash
	 *
	 * @{
	 *
	 **/

public:

	template<size_t IFORM> size_t max_hash() const
	{
		return topology_type::template max_hash<IFORM>(m_id_memory_min_,
				m_id_memory_max_);
	}
	constexpr size_t hash(id_type s) const
	{
		return topology_type::hash(s, m_id_memory_min_, m_id_memory_max_);
	}

	template<size_t IFORM>
	constexpr id_type pack_relative_index(index_type i, index_type j,
			index_type k, index_type n = 0) const
	{
		return topology_type::pack_index(nTuple<index_type, 3>( { i, j, k }),
				topology_type:: template sub_index_to_id<IFORM>(n)) + m_id_min_;
	}

	nTuple<index_type, ndims + 1> unpack_relative_index(id_type s) const
	{
		nTuple<index_type, ndims + 1> res;
		res = topology_type::unpack_index(s - m_id_min_);
		res[ndims] = topology_type::sub_index(s);
		return std::move(res);
	}

	/** @} */

	template<size_t IFORM> DataSpace dataspace() const
	{
		nTuple<index_type, ndims + 1> f_dims;
		nTuple<index_type, ndims + 1> f_offset;
		nTuple<index_type, ndims + 1> f_count;
		nTuple<index_type, ndims + 1> f_ghost_width;

		nTuple<index_type, ndims + 1> m_dims;
		nTuple<index_type, ndims + 1> m_offset;

		int f_ndims = ndims;

		f_dims = topology_type::unpack_index(m_id_max_ - m_id_min_);

		f_offset = topology_type::unpack_index(m_id_local_min_ - m_id_min_);

		f_count = topology_type::unpack_index(
				m_id_local_max_ - m_id_local_min_);

		m_dims = topology_type::unpack_index(
				m_id_memory_max_ - m_id_memory_min_);
		;

		m_offset = topology_type::unpack_index(m_id_local_min_ - m_id_min_);

		if ((IFORM == EDGE || IFORM == FACE))
		{
			f_ndims = ndims + 1;
			f_dims[ndims] = 3;
			f_offset[ndims] = 0;
			f_count[ndims] = 3;
			m_dims[ndims] = 3;
			m_offset[ndims] = 0;
		}
		else
		{
			f_ndims = ndims;
			f_dims[ndims] = 1;
			f_offset[ndims] = 0;
			f_count[ndims] = 1;
			m_dims[ndims] = 1;
			m_offset[ndims] = 0;
		}

		DataSpace res(f_ndims, &(f_dims[0]));

		res

		.select_hyperslab(&f_offset[0], nullptr, &f_count[0], nullptr)

		.set_local_shape(&m_dims[0], &m_offset[0]);

		return std::move(res);

	}
	template<size_t IFORM> void ghost_shape(
			std::vector<mpi_ghosts_shape_s> *res) const
	{
		nTuple<size_t, ndims + 1> f_local_dims;
		nTuple<size_t, ndims + 1> f_local_offset;
		nTuple<size_t, ndims + 1> f_local_count;
		nTuple<size_t, ndims + 1> f_ghost_width;

		int f_ndims = ndims;

//		f_local_dims = topology_type::unpack_index(
//				m_id_memory_max_ - m_id_memory_min_);

		f_local_offset = topology_type::unpack_index(
				m_id_local_min_ - m_id_memory_min_);

		f_local_count = topology_type::unpack_index(
				m_id_local_max_ - m_id_local_min_);

		f_ghost_width = topology_type::unpack_index(
				m_id_local_min_ - m_id_memory_min_);

		if ((IFORM == EDGE || IFORM == FACE))
		{
			f_ndims = ndims + 1;
//			f_local_dims[ndims] = 3;
			f_local_offset[ndims] = 0;
			f_local_count[ndims] = 3;
			f_ghost_width[ndims] = 0;
		}
		else
		{
			f_ndims = ndims;

//			f_local_dims[ndims] = 1;
			f_local_offset[ndims] = 0;
			f_local_count[ndims] = 1;
			f_ghost_width[ndims] = 0;

		}

		get_ghost_shape(f_ndims, &f_local_offset[0], nullptr, &f_local_count[0],
				nullptr, &f_ghost_width[0], res);

	}
	template<size_t IFORM> std::vector<mpi_ghosts_shape_s> ghost_shape() const
	{
		std::vector<mpi_ghosts_shape_s> res;
		ghost_shape<IFORM>(&res);
		return std::move(res);
	}

	template<typename DistanceFunction, typename Res>
	void select(DistanceFunction const & dist, int select_tag, int iform_tag,
			point_type const & x_min, point_type const & x_max, Res *res) const
	{

		id_type s = traits::get<0>(
				coordinates_global_to_local((x_min),
						topology_type::TAG_VERTEX));

		Vec3 L;
		L = inv_map(x_max) - inv_map(x_min);

		size_t level = static_cast<size_t>(std::log(
				traits::max(L[0], L[1], L[2])) / std::log(2.0)) + 1;

		size_t count = 0;

		topology_type::select(

		[&](id_type t)
		{
			++count;
			return static_cast<Real>( dist(map(topology_type::point(t))));
		}

		,

		[&](id_type t)
		{
			res->insert(t);
		},

		select_tag,

		iform_tag,

		s, level

		);

		CHECK(count);

	}

};

template<typename CoordinateSystem> constexpr size_t Mesh<CoordinateSystem,
		tags::structured>::DEFAULT_GHOST_WIDTH;

template<typename CoordinateSystem>
void Mesh<CoordinateSystem, tags::structured>::deploy(size_t const *gw)
{
	nTuple<id_type, ndims> dims = topology_type::unpack_index(
			m_id_max_ - m_id_min_);

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

			m_to_toplogy_orig_[i] = -m_coords_min_[i] * m_to_topology_scale_[i]
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

			m_to_toplogy_orig_[i] = -m_coords_min_[i] * m_to_topology_scale_[i]
					+ topology_type::INDEX_ZERO;

			m_from_topology_orig_[i] = -m_from_topology_scale_[i]
					* topology_type::INDEX_ZERO + m_coords_min_[i];
		}

	}

	m_id_max_ = m_id_min_ + topology_type::pack_index(dims);

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
	/**
	 * Decompose
	 */

	if (GLOBAL_COMM.num_of_process() > 1)
	{
		auto idx_b = topology_type::unpack_index(m_id_min_);

		auto idx_e = topology_type::unpack_index(m_id_max_);

		GLOBAL_COMM.decompose(ndims, &idx_b[0], &idx_e[0]);

		index_tuple ghost_width;

		if (gw != nullptr)
		{
			ghost_width = gw;
		}
		else
		{
			ghost_width = DEFAULT_GHOST_WIDTH;
		}

		for (int i = 0; i < ndims; ++i)
		{

			if (idx_b[i] + 1 == idx_e[i])
			{
				ghost_width[i] = 0;
			}
			else if (idx_e[i] <= idx_b[i] + ghost_width[i] * 2)
			{
				ERROR(
				"Dimension is to small to split!["
				" Dimensions= " + type_cast < std::string
				> (topology_type::unpack_index(
								m_id_max_ - m_id_min_))
				+ " , Local dimensions=" + type_cast
				< std::string
				> (topology_type::unpack_index(
								m_id_local_max_ - m_id_local_min_))
				+ " , Ghost width =" + type_cast
				< std::string > (ghost_width) + "]");
			}

		}

		m_id_local_min_ = topology_type::pack_index(idx_b);

		m_id_local_max_ = topology_type::pack_index(idx_e);

		m_id_memory_min_ = m_id_local_min_
		- topology_type::pack_index(ghost_width);

		m_id_memory_max_ = m_id_local_max_
		+ topology_type::pack_index(ghost_width);

		m_is_distributed_ = true;

	}
	else
	{
		m_id_local_min_ = m_id_min_;

		m_id_local_max_ = m_id_max_;

		m_id_memory_min_ = m_id_local_min_;

		m_id_memory_max_ = m_id_local_max_;

		m_is_distributed_ = false;
	}

	m_is_valid_ = true;

	VERBOSE<<get_type_as_string() << " is deployed!" << std::endl;

}

}
// namespace simpla

#endif /* CORE_MESH_STRUCTURED_H_ */
