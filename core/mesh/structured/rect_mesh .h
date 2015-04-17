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

#include "../../dataset/dataset.h"
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
struct RectMesh_: public MeshIDs_<NFLAGS...>
{

	typedef TCoordSystem coordinates_system;

	typedef RectMesh_<coordinates_system, NFLAGS...> this_type;

	typedef MeshIDs_<NFLAGS...> topology;

	using topology::ndims;

	using typename topology::index_type;

	using typename topology::index_tuple;

	using typename topology::id_type;

	using typename topology::coordinates_type;

private:
	bool is_valid_ = false;

//	index_tuple m_index_global_dimensions_ = { 1, 1, 1 };
//	index_tuple m_index_global_offset_ = { 0, 0, 0 };
//
//	index_tuple m_index_local_dimensions_ = { 1, 1, 1 };
//	index_tuple m_index_local_offset_ = { 0, 0, 0 };

	index_tuple m_index_global_begin_ = { 0, 0, 0 };
	index_tuple m_index_global_end_ = { 1, 1, 1 };

	index_tuple m_index_local_begin_ = { 0, 0, 0 };
	index_tuple m_index_local_end_ = { 1, 1, 1 };

	index_tuple m_index_ghost_width_ = { 0, 0, 0 };

	coordinates_type m_dx_ /*= { 0, 0, 0 }*/;

	coordinates_type m_coord_orig_ /*= { 0, 0, 0 }*/;

	coordinates_type m_coords_min_ = { 0, 0, 0 };

	coordinates_type m_coords_max_ = { 1, 1, 1 };

	coordinates_type m_inv_length_ /*= { 1.0, 1.0, 1.0 }*/;

	coordinates_type m_length_ /*= { 1.0, 1.0, 1.0 }*/;

	coordinates_type m_shift_/* = { 0, 0, 0 }*/;

	coordinates_type m_to_topology_factor_;

	coordinates_type m_from_topology_factor_;

//	coordinates_type m_xmin_, m_xmax_, m_dx_;

//	Real m_dt_ = 1.0;

public:

//***************************************************************************************************

	RectMesh_()
	{
	}

	~RectMesh_()
	{
	}

	RectMesh_(RectMesh_ const & other)
			:
//			m_xmin_(other.m_xmin_), m_xmax_(other.m_xmax_), m_dx_(other.m_dx_),

			m_index_global_dimensions_(other.m_index_global_dimensions_),

			m_index_global_offset_(other.m_index_global_offset_),

			m_index_local_dimensions_(other.m_index_local_dimensions_),

			m_index_local_offset_(other.m_index_local_offset_),

			m_index_count_(other.m_index_count_),

			m_index_ghost_width_(other.m_index_ghost_width_),

			m_index_grain_size_(other.m_index_grain_size_),

			m_dataspace_(other.m_dataspace_)
	{
	}

	void swap(RectMesh_ & other)
	{
//		std::swap(m_xmin_, other.m_xmin_);
//		std::swap(m_xmax_, other.m_xmax_);
//		std::swap(m_dx_, other.m_dx_);

		std::swap(m_index_global_dimensions_, other.m_index_global_dimensions_);
		std::swap(m_index_global_offset_, other.m_index_global_offset_);
		std::swap(m_index_local_dimensions_, other.m_index_local_dimensions_);
		std::swap(m_index_local_offset_, other.m_index_local_offset_);

		std::swap(m_index_count_, other.m_index_count_);
		std::swap(m_index_ghost_width_, other.m_index_ghost_width_);
		std::swap(m_index_grain_size_, other.m_index_grain_size_);
		std::swap(m_dataspace_, other.m_dataspace_);

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

		<< " Type = \"" << get_type_as_string() << "\", "

		<< " Dimensions =  " << dimensions()

		<< " Min = " << m_coords_min_ << " ," << std::endl

		<< " Max  = " << m_coords_max_ << "," << std::endl

		<< " dx  = " << m_dx_ << "," << std::endl

		<< " dt  = " << m_dt_ << "," << std::endl;

		return os;

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

	void deploy();

	template<typename T>
	void dimensions(T const& d)
	{
		m_index_global_dimensions_ = d;
	}

	auto dimensions() const
	DECL_RET_TYPE(m_index_global_dimensions_)

	template<typename T>
	void ghost_width(T const&d)
	{
		m_index_ghost_width_ = d;
	}

	auto ghost_width() const
	DECL_RET_TYPE(m_index_ghost_width_)

	template<typename T0, typename T1>
	void extents(T0 const& pmin, T1 const& pmax)
	{
		m_coords_min_ = pmin;
		m_coords_max_ = pmax;
	}
	std::pair<coordinates_type, coordinates_type> extents() const
	{
		coordinates_type xmin;
		coordinates_type xmax;
		xmin = 0;
		xmax = m_index_global_dimensions_;
		return std::make_pair(xmin, xmax);
	}
	std::pair<coordinates_type, coordinates_type> local_extents() const
	{
		coordinates_type xmin;
		coordinates_type xmax;
		xmin = m_index_min_;
		xmax = m_index_max_;
		return std::make_pair(xmin, xmax);
	}

	typename topology::domain_type domain() const
	{
		return typename topology::domain_type(m_index_global_offset_,
				m_index_global_offset_ + m_index_count_);
	}

	typename topology::domain_type global_domain() const
	{
		return typename topology::domain_type(0, m_index_global_dimensions_);
	}

	inline auto extents() const
	DECL_RET_TYPE (std::make_pair(m_coords_min_, m_coords_max_))

	coordinates_type const & dx() const
	{
		return m_dx_;
	}
	void dx(coordinates_type const & pdx)
	{
		m_dx_ = pdx;
	}
	/**
	 * @name  Coordinates map
	 * @{
	 **/
	coordinates_type coordinates(id_type const & s) const
	{
		return std::move(
				coordinates_from_topology(topology::id_to_coordinates(s)));
	}

	template<size_t IFORM = 0>
	inline id_type coordinates_to_id(coordinates_type const &x, int n = 0) const
	{
		return topology::coordinates_to_id<IFORM>(coordinates_to_topology(x), n);
	}

	coordinates_type coordinates_from_topology(coordinates_type const &y) const
	{

		return coordinates_type(
				{

				std::fma(y[0] - topology::COORD_ZERO,
						m_from_topology_factor_[0], m_coord_orig_[0]),

				std::fma(y[1] - topology::COORD_ZERO,
						m_from_topology_factor_[1], m_coord_orig_[1]),

				std::fma(y[2] - topology::COORD_ZERO,
						m_from_topology_factor_[2], m_coord_orig_[2])

				});

	}
	coordinates_type coordinates_to_topology(coordinates_type const &x) const
	{
		return coordinates_type(
				{

				std::fma(x[0] - m_coord_orig_[0], m_to_topology_factor_[0],
						topology::COORD_ZERO),

				std::fma(x[1] - m_coord_orig_[1], m_to_topology_factor_[1],
						topology::COORD_ZERO),

				std::fma(x[2] - m_coord_orig_[2], m_to_topology_factor_[2],
						topology::COORD_ZERO)

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
						topology::coordinates_local_to_global(
								std::forward<Args >(args)...)));
	}

	template<size_t IFORM>
	std::tuple<id_type, coordinates_type> coordinates_global_to_local(
			coordinates_type x, int n = 0) const
	{
		return std::move(
				topology::template coordinates_global_to_local<IFORM>(
						std::move(coordinates_to_topology(x)), n));
	}

	/** @} */
	/**
	 * 	@name  Time
	 *  @{
	 *
	 */

private:
	Real m_time_ = 0;
	Real m_dt_ = 1.0;

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
	/**
	 * @name  Data Shape
	 * @{
	 **/

	template<size_t IFORM = VERTEX>
	DataSpace dataspace() const
	{
		nTuple<size_t, MAX_NDIMS_OF_ARRAY> f_dims;
		nTuple<size_t, MAX_NDIMS_OF_ARRAY> f_offset;
		nTuple<size_t, MAX_NDIMS_OF_ARRAY> f_count;
		nTuple<size_t, MAX_NDIMS_OF_ARRAY> f_ghost_width;
		int f_ndims = ndims;
		f_dims = m_index_global_end_ - m_index_global_begin_;
		f_offset = m_index_local_begin_ - m_index_global_begin_;
		f_count = m_index_local_end_ - m_index_local_begin_;
		f_ghost_width = m_index_ghost_width_;

		if ((IFORM != VERTEX && IFORM != VOLUME))
		{
			f_ndims = ndims + 1;
			f_dims[f_ndims - 1] = 3;
			f_offset[f_ndims - 1] = 0;
			f_count[f_ndims - 1] = 3;
			f_ghost_width[f_ndims - 1] = 0;
		}

		DataSpace res(f_ndims, &f_dims[0]);

		res

		.select_hyperslab(&f_offset[0], nullptr, &f_count[0], nullptr)

		.add_ghosts(&f_ghost_width[0]);

		return std::move(res);

	}

	template<size_t IFORM = VERTEX>
	void ghost_shape(std::vector<mpi_ghosts_shape_s> *res) const
	{
		nTuple<size_t, MAX_NDIMS_OF_ARRAY> f_dims;
		nTuple<size_t, MAX_NDIMS_OF_ARRAY> f_offset;
		nTuple<size_t, MAX_NDIMS_OF_ARRAY> f_count;
		nTuple<size_t, MAX_NDIMS_OF_ARRAY> f_ghost_width;
		int f_ndims = ndims;
		f_dims = m_index_global_end_ - m_index_global_begin_;
		f_offset = m_index_local_begin_ - m_index_global_begin_;
		f_count = m_index_local_end_ - m_index_local_begin_;
		f_ghost_width = m_index_ghost_width_;

		if ((IFORM != VERTEX && IFORM != VOLUME))
		{
			f_ndims = ndims + 1;
			f_dims[ndims] = 3;
			f_offset[ndims] = 0;
			f_count[ndims] = 3;
			f_ghost_width[ndims] = 0;
		}

		get_ghost_shape(f_ndims, &f_dims[0], &f_offset[0], nullptr, &f_count[0],
				nullptr, &f_ghost_width[0], res);

	}
	template<size_t IFORM = VERTEX>
	std::vector<mpi_ghosts_shape_s> ghost_shape() const
	{
		std::vector<mpi_ghosts_shape_s> res;
		ghost_shape<IFORM>(&res);
		return std::move(res);
	}
	/** @}*/

	/**
	 *  @name Hash
	 *  @{
	 */

	template<size_t IFORM = VERTEX>
	constexpr size_t max_hash() const
	{
		return m_hasher_.template max_hash<IFORM>();
	}

	template<size_t IFORM = VERTEX>
	constexpr size_t hash(id_type const &s) const
	{
		return m_hasher_.template hash<IFORM>(s);
	}

	template<size_t IFORM = VERTEX>
	MeshIDs::id_hasher<IFORM> hasher() const
	{
		return MeshIDs::id_hasher<IFORM>();
	}

	/**@}*/

	/** @name Volume
	 * @{
	 */
	typename topology::volume_container m_volume_;

	typename topology::volume_container m_dual_volume_;

	void deploy_volume()
	{

		m_volume_ = make_valume_container(m_index_local_dimensions_ * 2,
				(m_index_global_offset_ + m_index_local_offset_) * 2);

		m_dual_volume_ = make_valume_container(m_index_local_dimensions_ * 2,
				(m_index_global_offset_ + m_index_local_offset_) * 2);

		m_volume_.deploy();
		m_dual_volume_.deploy();

	}

	Real const &volume(id_type s) const
	{
		return m_volume_[s];
	}

	Real const &dual_volume(id_type s) const
	{
		return m_dual_volume_[s];
	}

	constexpr Real cell_volume(id_type s) const
	{
		return volume(s | MeshIDs::_DA);
	}

	constexpr Real inv_volume(id_type s) const
	{
		return 1.0 / volume(s);
	}

	constexpr Real inv_dual_volume(id_type s) const
	{
		return 1.0 / dual_volume(s);;
	}
	/**@}*/

}
;

template<typename TCoord, size_t ... N>
void RectMesh_<TCoord, N...>::deploy()
{

	m_index_count_ = m_index_global_dimensions_;

	m_index_global_offset_ = 0;

	if (GLOBAL_COMM.num_of_process() > 1)
	{
		GLOBAL_COMM.decompose(ndims, &m_index_global_offset_[0],
		&m_index_count_[0]);
	}

	m_index_min_ = m_index_global_offset_;
	m_index_max_ = m_index_min_ + m_index_count_;

	m_index_local_offset_ = m_index_ghost_width_;
	m_index_local_dimensions_ = m_index_count_ + m_index_ghost_width_ * 2;

	is_valid_ = true;

	typename topology::template id_hasher<0>(m_index_local_dimensions_,
			m_index_global_offset_ - m_index_local_offset_).swap(m_hasher_);

	VERBOSE << get_type_as_string() << " is deployed!" << std::endl;
	index_tuple dims;

	for (size_t i = 0; i < ndims; ++i)
	{
		if (m_dx_[i] > EPSILON
				&& (m_coords_max_[i] - m_coords_min_[i]) > m_dx_[i])
		{
			dims[i] = static_cast<size_t>((m_coords_max_[i] - m_coords_min_[i])
					/ m_dx_[i]);

			m_dx_[i] = (m_coords_max_[i] - m_coords_min_[i])
					/ static_cast<Real>(dims[i]);
		}
		else
		{
			m_dx_[i] = 0;
			m_coords_max_[i] = m_coords_min_[i];
			dims[i] = 1;
		}
	}
	for (size_t i = 0; i < ndims; ++i)
	{
		m_shift_[i] = m_coords_min_[i];

		if (dims[i] <= 1)
		{

			m_coords_max_[i] = m_coords_min_[i];

			m_inv_length_[i] = 0.0;

			m_length_[i] = 0.0;

			volume_[1UL << (ndims - i - 1)] = 1.0;

			dual_volume_[7 - (1UL << (ndims - i - 1))] = 1.0;

			inv_volume_[1UL << (ndims - i - 1)] = 1.0;

			inv_dual_volume_[7 - (1UL << (ndims - i - 1))] = 1.0;

			m_to_topology_factor_[i] = 0;
			m_from_topology_factor_[i] = 0;
		}
		else
		{
			m_inv_length_[i] = 1.0 / (m_coords_max_[i] - m_coords_min_[i]);

			m_length_[i] = (m_coords_max_[i] - m_coords_min_[i]);

			volume_[1UL << (ndims - i - 1)] = m_length_[i];

			dual_volume_[7 - (1UL << (ndims - i - 1))] = m_length_[i];

			inv_volume_[1UL << (ndims - i - 1)] = m_inv_length_[i];

			inv_dual_volume_[7 - (1UL << (ndims - i - 1))] = m_inv_length_[i];

			m_to_topology_factor_ = dims[i]
					/ (m_coords_max_[i] - m_coords_min_[i]);

			m_from_topology_factor_[i] = (m_coords_max_[i] - m_coords_min_[i])
					/ dims[i];

		}
	}

	DEFINE_PHYSICAL_CONST

	auto dx_ = dx();

	Real safe_dt = m_CFL_
			* std::sqrt(dx_[0] * dx_[0] + dx_[1] * dx_[1] + dx_[2] * dx_[2])
			/ speed_of_light;

	if (m_dt_ > safe_dt)
	{
		WARGING("  Courant–Friedrichs–Lewy (CFL) condition error  ");
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
//	volume_[0] = 1;
////		volume_[1] /* 001 */= dx_[0];
////		volume_[2] /* 010 */= dx_[1];
////		volume_[4] /* 100 */= dx_[2];
//
//	volume_[3] /* 011 */= volume_[1] * volume_[2];
//	volume_[5] /* 101 */= volume_[4] * volume_[1];
//	volume_[6] /* 110 */= volume_[2] * volume_[4];
//
//	volume_[7] /* 111 */= volume_[1] * volume_[2] * volume_[4];
//
//	dual_volume_[7] = 1;
////		dual_volume_[6] /* 001 */= dx_[0];
////		dual_volume_[5] /* 010 */= dx_[1];
////		dual_volume_[3] /* 100 */= dx_[2];
//
//	dual_volume_[4] /* 011 */= dual_volume_[6] * dual_volume_[5];
//	dual_volume_[2] /* 101 */= dual_volume_[3] * dual_volume_[6];
//	dual_volume_[1] /* 110 */= dual_volume_[5] * dual_volume_[3];
//
//	dual_volume_[0] /* 111 */= dual_volume_[6] * dual_volume_[5]
//			* dual_volume_[3];
//
//	inv_volume_[0] = 1;
////		inv_volume_[1] /* 001 */= inv_dx_[0];
////		inv_volume_[2] /* 010 */= inv_dx_[1];
////		inv_volume_[4] /* 100 */= inv_dx_[2];
//
//	inv_volume_[3] /* 011 */= inv_volume_[1] * inv_volume_[2];
//	inv_volume_[5] /* 101 */= inv_volume_[4] * inv_volume_[1];
//	inv_volume_[6] /* 110 */= inv_volume_[2] * inv_volume_[4];
//
//	inv_volume_[7] /* 111 */= inv_volume_[1] * inv_volume_[2] * inv_volume_[4];
//
//	inv_dual_volume_[7] = 1;
////		inv_dual_volume_[6] /* 001 */= inv_dx_[0];
////		inv_dual_volume_[5] /* 010 */= inv_dx_[1];
////		inv_dual_volume_[3] /* 100 */= inv_dx_[2];
//
//	inv_dual_volume_[4] /* 011 */= inv_dual_volume_[6] * inv_dual_volume_[5];
//	inv_dual_volume_[2] /* 101 */= inv_dual_volume_[3] * inv_dual_volume_[6];
//	inv_dual_volume_[1] /* 110 */= inv_dual_volume_[5] * inv_dual_volume_[3];
//
//	inv_dual_volume_[0] /* 111 */= inv_dual_volume_[6] * inv_dual_volume_[5]
//			* inv_dual_volume_[3];
//
//	updatedt();
//
//	is_valid_ = true;

	VERBOSE << get_type_as_string() << " is deployed!" << std::endl;

}
}
// namespace simpla

//}  // namespace std

#endif /* MESH_RECT_MESH_H_ */
