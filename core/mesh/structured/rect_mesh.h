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
//template<typename ... >struct RectMesh;
template<typename TTopology, typename ...Policies>
struct RectMesh: public TTopology, public Policies...
{
	typedef TTopology topology_type;
	typedef typename unpack_typelist<0, Policies...>::type coordinates_system;
	typedef typename unpack_typelist<1, Policies...>::type interpolatpr_policy;
	typedef typename unpack_typelist<2, Policies...>::type calculate_policy;

	typedef RectMesh<topology_type, Policies...> this_type;

	using topology_type::ndims;

	using typename topology_type::index_type;

	using typename topology_type::index_tuple;

	using typename topology_type::id_type;

	using typename topology_type::coordinates_type;

	friend class Domain<this_type, VERTEX> ;
	friend class Domain<this_type, EDGE> ;
	friend class Domain<this_type, FACE> ;
	friend class Domain<this_type, VOLUME> ;

private:

	static constexpr size_t DEFAULT_GHOST_WIDTH = 2;

	bool m_is_valid_ = false;

	bool m_is_distributed_ = false;

	coordinates_type m_coord_orig_ /*= { 0, 0, 0 }*/;

	coordinates_type m_toplogy_coord_orig_ /*= { 0, 0, 0 }*/;

	coordinates_type m_coords_min_ = { 0, 0, 0 };

	coordinates_type m_coords_max_ = { 1, 1, 1 };

	coordinates_type m_dx_ /*= { 0, 0, 0 }*/;

	coordinates_type m_to_topology_scale_;

	coordinates_type m_from_topology_scale_;

	/**
	 *
	 *   a----------------------------b
	 *   |                            |
	 *   |     c--------------d       |
	 *   |     |              |       |
	 *   |     |  e*******f   |       |
	 *   |     |  *       *   |       |
	 *   |     |  *       *   |       |
	 *   |     |  *       *   |       |
	 *   |     |  *********   |       |
	 *   |     ----------------       |
	 *   ------------------------------
	 *
	 *   a=0
	 *   b-a = dimension
	 *   e-a = offset
	 *   f-e = count
	 *   d-c = local_dimension
	 *   c-a = local_offset
	 */
	index_tuple m_index_dimensions_ = { 1, 1, 1 };

	index_tuple m_index_offset_ = { 0, 0, 0 };
	index_tuple m_index_count_ = { 1, 1, 1 };

	index_tuple m_index_local_dimensions_ = { 0, 0, 0 };
	index_tuple m_index_local_offset_ = { 0, 0, 0 };

public:

//***************************************************************************************************

	RectMesh()
	{
	}

	~RectMesh()
	{
	}

	RectMesh(this_type const & other) :

			m_index_dimensions_(other.m_index_dimensions_),

			m_index_offset_(other.m_index_offset_),

			m_index_count_(other.m_index_count_),

			m_index_local_dimensions_(other.m_index_local_dimensions_),

			m_index_local_offset_(other.m_index_local_offset_),

			m_hash_strides_(other.m_hash_strides_)

	{
	}

	void swap(this_type & other)
	{
		std::swap(m_index_dimensions_, other.m_index_dimensions_);
		std::swap(m_index_offset_, other.m_index_offset_);
		std::swap(m_index_count_, other.m_index_count_);

		std::swap(m_index_local_dimensions_, other.m_index_local_dimensions_);
		std::swap(m_index_local_offset_, other.m_index_local_offset_);

		std::swap(m_hash_strides_, other.m_hash_strides_);
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

		<< " Min \t= " << m_coords_min_ << " ," << std::endl

		<< " Max \t= " << m_coords_max_ << "," << std::endl

		<< " dx  \t= " << m_dx_ << "," << std::endl

		<< " dt \t= " << m_dt_ << "," << std::endl

		<< " Dimensionss \t= " << m_index_dimensions_ << "," << std::endl

		;

		return os;

	}

	static std::string get_type_as_string()
	{
		return "RectMesh<" + coordinates_system::get_type_as_string() + ">";
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

	inline auto extents() const
	DECL_RET_TYPE (std::make_pair(m_coords_min_, m_coords_max_))

	coordinates_type const & dx() const
	{
		return m_dx_;
	}

	template<typename TI> void dimensions(TI const & d)
	{
		m_index_dimensions_ = d;
	}
	index_tuple dimensions() const
	{
		return m_index_dimensions_;
	}

	void deploy(size_t const *gw = nullptr);

	template<size_t IFORM>
	Domain<this_type, IFORM> domain() const
	{
		return Domain<this_type, IFORM>(*this);
	}

	template<size_t IFORM, typename ...Args>
	Domain<this_type, IFORM> domain(Args &&... args) const
	{
		return Domain<this_type, IFORM>(*this, std::forward<Args>(args)...);
	}

	template<size_t IFORM, typename ...Args>
	auto sample(Args && ...args) const
	DECL_RET_TYPE( interpolatpr_policy:: template sample<IFORM>(
					*this ,std::forward<Args>(args)...))

	template<typename ...Args>
	auto calculate(Args && ...args) const
	DECL_RET_TYPE((calculate_policy:: calculate(
							*this,std::forward<Args>(args)...)))

	template<typename ...Args>
	auto gather(Args && ...args) const
	DECL_RET_TYPE((interpolatpr_policy::gather(
							*this,std::forward<Args>(args)...)))

	template<typename ...Args>
	auto scatter(Args && ...args) const
	DECL_RET_TYPE((interpolatpr_policy:: scatter(
							*this,std::forward<Args>(args)...)))

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
		return topology_type::template coordinates_to_id<IFORM>(
				coordinates_to_topology(x), n);
	}

	template<size_t IFORM = VERTEX>
	inline index_tuple coordinates_to_index(coordinates_type const &x) const
	{
		return topology_type::template coordinates_to_index<IFORM>(
				coordinates_to_topology(x));
	}

	coordinates_type coordinates_from_topology(coordinates_type const &y) const
	{

		return coordinates_type( {

		std::fma(y[0], m_from_topology_scale_[0], m_coord_orig_[0]),

		std::fma(y[1], m_from_topology_scale_[1], m_coord_orig_[1]),

		std::fma(y[2], m_from_topology_scale_[2], m_coord_orig_[2])

		});

	}
	coordinates_type coordinates_to_topology(coordinates_type const &x) const
	{

		return coordinates_type( {

		std::fma(x[0], m_to_topology_scale_[0], m_toplogy_coord_orig_[0]),

		std::fma(x[1], m_to_topology_scale_[1], m_toplogy_coord_orig_[1]),

		std::fma(x[2], m_to_topology_scale_[2], m_toplogy_coord_orig_[2])

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

	/**@name hash
	 *
	 * @{
	 *
	 **/
private:
	size_t m_hash_max_ = 0;

	index_tuple m_hash_strides_;

public:

	template<size_t IFORM>
	size_t max_hash() const
	{
		return m_hash_max_ * ((IFORM == EDGE || IFORM == FACE) ? 3 : 1);
	}
	template<size_t IFORM>
	constexpr size_t hash(id_type const &s) const
	{
		return inner_product(
				(m_index_local_dimensions_
						+ topology_type::template unpack<IFORM>(s)
						- m_index_local_offset_) % m_index_local_dimensions_,
				m_hash_strides_) * ((IFORM == EDGE || IFORM == FACE) ? 3 : 1)
				+ topology_type::node_id(s);
	}

	/** @} */

	template<size_t IFORM>
	DataSpace dataspace() const
	{
		nTuple<index_type, ndims + 1> f_dims;
		nTuple<index_type, ndims + 1> f_offset;
		nTuple<index_type, ndims + 1> f_count;
		nTuple<index_type, ndims + 1> f_ghost_width;

		int f_ndims = ndims;

		f_dims = m_index_dimensions_;

		f_offset = m_index_offset_;

		f_count = m_index_count_;

		f_ghost_width = m_index_offset_ - m_index_local_offset_;

		if ((IFORM == EDGE || IFORM == FACE))
		{
			f_ndims = ndims + 1;
			f_dims[ndims] = 3;
			f_offset[ndims] = 0;
			f_count[ndims] = 3;
		}

		f_ghost_width[ndims] = 0;

		DataSpace res(f_ndims, &(f_dims[0]));

		res

		.select_hyperslab(&f_offset[0], nullptr, &f_count[0], nullptr)

		.convert_to_local(&f_ghost_width[0]);

		return std::move(res);

	}
	template<size_t IFORM>
	void ghost_shape(std::vector<mpi_ghosts_shape_s> *res) const
	{
		nTuple<size_t, ndims + 1> f_dims;
		nTuple<size_t, ndims + 1> f_offset;
		nTuple<size_t, ndims + 1> f_count;
		nTuple<size_t, ndims + 1> f_ghost_width;

		int f_ndims = ndims;

		f_dims = m_index_local_dimensions_;

		f_count = m_index_count_;

		f_offset = m_index_offset_ - m_index_local_offset_;

		f_ghost_width = f_offset;

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
	template<size_t IFORM>
	std::vector<mpi_ghosts_shape_s> ghost_shape() const
	{
		std::vector<mpi_ghosts_shape_s> res;
		ghost_shape<IFORM>(&res);
		return std::move(res);
	}
}
;

template<typename TTopology, typename ... Polices>
constexpr size_t RectMesh<TTopology, Polices...>::DEFAULT_GHOST_WIDTH;

template<typename TTopology, typename ... Polices>
void RectMesh<TTopology, Polices...>::deploy(size_t const *gw)
{
	CHECK(m_index_dimensions_);
	for (int i = 0; i < ndims; ++i)
	{
		if (m_index_dimensions_[i] > 0
				&& (m_coords_max_[i] - m_coords_min_[i]) > EPSILON)
		{

			m_dx_[i] = (m_coords_max_[i] - m_coords_min_[i])
					/ static_cast<Real>(m_index_dimensions_[i]);

			m_to_topology_scale_[i] = static_cast<Real>(m_index_dimensions_[i])
					/ (m_coords_max_[i] - m_coords_min_[i]);

			m_from_topology_scale_[i] = (m_coords_max_[i] - m_coords_min_[i])
					/ static_cast<Real>(m_index_dimensions_[i]);
		}
#ifdef  ENABLE_COMPLEX_COORDINATE_SYSTEM
		else if ((m_coords_max_[i] - m_coords_min_[i]) > EPSILON)
		{
			m_index_dimensions_[i] = 1;
			m_dx_[i] = 0;
			m_to_topology_scale_ = 1.0 / (m_coords_max_[i] - m_coords_min_[i]);
			m_from_topology_scale_[i] = (m_coords_max_[i] - m_coords_min_[i])
			/ 1.0;
		}
#endif
		else
		{
			m_index_dimensions_[i] = 1;

			m_dx_[i] = 0;

			m_coords_max_[i] = m_coords_min_[i];

			m_to_topology_scale_[i] = 0;
			m_from_topology_scale_[i] = 0;
		}

	}

	m_coord_orig_ = (m_coords_max_ + m_coords_min_) * 0.5;

	m_toplogy_coord_orig_ = -(m_coords_max_ + m_coords_min_) * 0.5
			* m_to_topology_scale_;

	DEFINE_PHYSICAL_CONST

	Real safe_dt = m_CFL_
			* std::sqrt(
					m_dx_[0] * m_dx_[0] + m_dx_[1] * m_dx_[1]
							+ m_dx_[2] * m_dx_[2]) / speed_of_light;

	if (m_dt_ > safe_dt)
	{
		WARNING << ("  Courant–Friedrichs–Lewy (CFL) !") << std::endl;
	}

	/**
	 *  deploy volume
	 **/

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

	/**
	 * Decompose
	 */

	m_index_count_ = m_index_dimensions_;

	m_index_offset_ = 0;

	if (GLOBAL_COMM.num_of_process() > 1)
	{
		GLOBAL_COMM.decompose(ndims, &m_index_count_[0], &m_index_offset_[0]);

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

			if (m_index_count_[i] == m_index_dimensions_[i])
			{
				ghost_width[i] = 0;
			}
			else if (m_index_count_[i] <= ghost_width[i] * 2)
			{
				ERROR(
				"Dimension is to small to split!["
				" Dimensions= "
				+ value_to_string(m_index_dimensions_)
				+ " , Local dimensions="
				+ value_to_string(m_index_count_)
				+ " , Ghost width ="
				+ value_to_string(ghost_width) + "]");
			}

		}

		m_index_local_dimensions_ = m_index_count_ + ghost_width * 2;

		m_index_local_offset_ = m_index_offset_ - ghost_width;

		m_is_distributed_ = true;

	}
	else
	{
		m_index_local_dimensions_ = m_index_count_;

		m_index_local_offset_ = m_index_offset_;

		m_is_distributed_ = false;
	}

	/**
	 *  Hash
	 */

	m_hash_strides_[ndims - 1] = 1;

	for (int i = ndims - 2; i >= 0; --i)
	{
		m_hash_strides_[i] = m_hash_strides_[i + 1]
				* m_index_local_dimensions_[i + 1];
	}
	m_hash_max_ = m_hash_strides_[0] * m_index_local_dimensions_[0];

	m_is_valid_ = true;

	VERBOSE << get_type_as_string() << " is deployed!" << std::endl;

}

}
// namespace simpla

#endif /* MESH_RECT_MESH_H_ */
