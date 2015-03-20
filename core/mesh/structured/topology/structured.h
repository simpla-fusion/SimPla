/**
 * @file  structured.h
 *
 *  created on: 2014-2-21
 *      Author: salmon
 */

#ifndef MESH_STRUCTURED_H_
#define MESH_STRUCTURED_H_

#include <stddef.h>
#include <algorithm>
#include <functional>
#include <iterator>
#include <limits>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>

#include "../../../dataset/dataset.h"
#include "../../../utilities/utilities.h"
#include "../../../gtl/ntuple.h"
#include "../../../gtl/primitives.h"
#include "../../../gtl/containers/sp_hash_container.h"

#include "../../../parallel/mpi_comm.h"
#include "../../../parallel/mpi_aux_functions.h"
#include "../../mesh_ids.h"

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
template<size_t NDIMS = 3, size_t AXIS_FLAG>
struct StructuredMesh_: public MeshIDs_<NDIMS, AXIS_FLAG>
{

	typedef StructuredMesh_ this_type;

	typedef MeshIDs_<NDIMS> ids;

	using ids::ndims;
	using typename ids::index_type;
	using typename ids::id_type;
	using typename ids::index_tuple;
	using typename ids::coordinates_type;

	static constexpr size_t DEFAULT_GHOSTS_WIDTH = 3;
	static constexpr size_t iform = VERTEX;

private:
	bool is_valid_ = false;

	index_tuple m_index_global_dimensions_ = { 1, 1, 1 };
	index_tuple m_index_global_offset_ = { 0, 0, 0 };

	index_tuple m_index_local_dimensions_ = { 1, 1, 1 };
	index_tuple m_index_local_offset_ = { 0, 0, 0 };

	index_tuple m_index_count_ = { 1, 1, 1 };
	index_tuple m_index_ghost_width_ = { 0, 0, 0 };
	index_tuple m_index_grain_size_;

	index_tuple m_index_min_ = { 1, 1, 1 };
	index_tuple m_index_max_ = { 0, 0, 0 };

//	coordinates_type m_xmin_, m_xmax_, m_dx_;

//	Real m_dt_ = 1.0;

	DataSpace m_dataspace_;

	std::vector<mpi_ghosts_shape_s> m_ghosts_shape_;

public:

	//***************************************************************************************************

	StructuredMesh_()
	{
	}

	StructuredMesh_(nTuple<size_t, ndims> const &dims)
	{
		dimensions(&dims[0]);
		m_index_ghost_width_ = 0;
	}
	virtual ~StructuredMesh_()
	{
	}

	StructuredMesh_(StructuredMesh_ const & other)
			:
//			m_xmin_(other.m_xmin_), m_xmax_(other.m_xmax_), m_dx_(other.m_dx_),

			m_index_global_dimensions_(other.m_index_global_dimensions_),

			m_index_global_offset_(other.m_index_global_offset_),

			m_index_local_dimensions_(other.m_index_local_dimensions_),

			m_index_local_offset_(other.m_index_local_offset_),

			m_index_local_strides_(other.m_index_local_strides_),

			m_index_count_(other.m_index_count_),

			m_index_ghost_width_(other.m_index_ghost_width_),

			m_index_grain_size_(other.m_index_grain_size_),

			m_dataspace_(other.m_dataspace_)
	{
	}

	void swap(StructuredMesh_ & other)
	{
//		std::swap(m_xmin_, other.m_xmin_);
//		std::swap(m_xmax_, other.m_xmax_);
//		std::swap(m_dx_, other.m_dx_);

		std::swap(m_index_global_dimensions_, other.m_index_global_dimensions_);
		std::swap(m_index_global_offset_, other.m_index_global_offset_);
		std::swap(m_index_local_dimensions_, other.m_index_local_dimensions_);
		std::swap(m_index_local_offset_, other.m_index_local_offset_);
		std::swap(m_index_local_strides_, other.m_index_local_strides_);

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

	template<typename TDict>
	bool load(TDict const & dict)
	{
		if (dict["Type"].template as<std::string>("") != get_type_as_string())
		{
			WARNING
					<< "Illegal topology type! "
							+ dict["Type"].template as<std::string>();
		}

		if (dict["Dimensions"].is_table())
		{
			VERBOSE << "Load topology : Structured  Mesh " << std::endl;
			auto d = dict["Dimensions"].template as<nTuple<size_t, 3>>();
			dimensions(d);

		}

		return true;

	}

	template<typename OS>
	OS & print(OS &os) const
	{
		os

		<< " Type = \"" << get_type_as_string() << "\", "

		<< " Dimensions =  " << dimensions()

		;

		return os;
	}
	static std::string get_type_as_string()
	{
		return "StructuredMesh";
	}

	constexpr bool is_valid() const
	{
		return is_valid_;
	}
	constexpr bool is_divisible() const
	{
		return false;
	}

	/**
	 * @name  Data Shape
	 * @{
	 **/

public:
	template<typename TV, size_t IFORM = iform>
	using field_value_type=typename std::conditional<IFORM==EDGE ||IFORM==FACE,nTuple<TV,3>,TV>::type;

private:
	Real m_time_ = 0;
public:
	void next_timestep()
	{
		m_time_ += 1.0;
	}
	void time(Real p_time)
	{
		m_time_ = p_time;
	}
	Real time() const
	{
		return m_time_;
	}

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

//	void dt(Real pdt)
//	{
//		m_dt_ = pdt;
//	}
	Real dt() const
	{
		return 1.0; //m_dt_;
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

	std::pair<index_tuple, index_tuple> range() const
	{
		return std::make_pair(m_index_min_, m_index_max_);
	}

	void deploy()
	{

		m_index_count_ = m_index_global_dimensions_;
		m_index_global_offset_ = 0;
		if (GLOBAL_COMM.num_of_process()>1)
		{
			GLOBAL_COMM.decompose(ndims, &m_index_global_offset_[0], &m_index_count_[0]);
		}

		m_index_min_ = m_index_global_offset_;
		m_index_max_ = m_index_min_ + m_index_count_;

		m_index_local_offset_ = m_index_ghost_width_;
		m_index_local_dimensions_ = m_index_count_ + m_index_ghost_width_ * 2;

		is_valid_ = true;

		VERBOSE << get_type_as_string() << " is deployed!" << std::endl;

	}
	template<size_t IFORM = iform>
	DataSpace dataspace() const
	{
		nTuple<size_t, MAX_NDIMS_OF_ARRAY> f_dims;
		nTuple<size_t, MAX_NDIMS_OF_ARRAY> f_offset;
		nTuple<size_t, MAX_NDIMS_OF_ARRAY> f_count;
		nTuple<size_t, MAX_NDIMS_OF_ARRAY> f_ghost_width;
		int f_ndims = ndims;
		f_dims = m_index_global_dimensions_;
		f_offset = m_index_global_offset_;
		f_count = m_index_count_;
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

		.convert_to_local(&f_ghost_width[0]);

		return std::move(res);

	}

	template<size_t IFORM = iform>
	void ghost_shape(std::vector<mpi_ghosts_shape_s> *res) const
	{
		nTuple<size_t, MAX_NDIMS_OF_ARRAY> f_dims;
		nTuple<size_t, MAX_NDIMS_OF_ARRAY> f_offset;
		nTuple<size_t, MAX_NDIMS_OF_ARRAY> f_count;
		nTuple<size_t, MAX_NDIMS_OF_ARRAY> f_ghost_width;
		int f_ndims = ndims;
		f_dims = m_index_local_dimensions_;
		f_offset = m_index_local_offset_;
		f_count = m_index_count_;
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
	template<size_t IFORM = iform>
	std::vector<mpi_ghosts_shape_s> ghost_shape() const
	{
		std::vector<mpi_ghosts_shape_s> res;
		ghost_shape<IFORM>(&res);
		return std::move(res);
	}

	typename MeshIDs::volume_container m_volume_;

	typename MeshIDs::volume_container m_dual_volume_;

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
		return m_volume_[MeshIDs::dual(s)];
	}

	/**
	 *  @name Hash
	 *  @{
	 */

	MeshIDs::id_hasher<MeshIDs::FLOATING_POINT_POS> m_hasher_;

	template<size_t IFORM = iform>
	constexpr size_t max_hash() const
	{
		return m_hasher_.template max_hash<IFORM>();
	}

	template<size_t IFORM = VERTEX>
	constexpr size_t hash(id_type const &s) const
	{
		return m_hasher_.template hash<IFORM>(s);
	}

	auto const &hasher() const
	{
		return m_hasher_;
	}

	/**@}*/

	template<typename TD>
	auto gather(TD const & d,
			coordinates_type const & x) const->decltype(d[std::declval<index_tuple>()])
	{
		index_tuple r;
		r = (x + 0.5);

		return d[r];
	}

	template<typename TD, typename TV>
	void scatter(TD & d, coordinates_type const &x, TV const & v) const
	{
		index_tuple r;
		r = (x + 0.5);

		d[r] += v;
	}

}
;

template<size_t N, size_t A> constexpr size_t StructuredMesh_<N, A>::DEFAULT_GHOSTS_WIDTH;
template<size_t N, size_t A> constexpr size_t StructuredMesh_<N, A>::iform;

typedef StructuredMesh_<3, 0> StructuredMesh;
}
// namespace simpla

//}  // namespace std

#endif /* MESH_STRUCTURED_H_ */
