/*
 * structured.h
 *
 *  created on: 2014-2-21
 *      Author: salmon
 */

#ifndef STRUCTURED_H_
#define STRUCTURED_H_

#include <stddef.h>
#include <algorithm>
#include <functional>
#include <iterator>
#include <limits>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>

#include "../../utilities/log.h"
#include "../../utilities/ntuple.h"
#include "../../utilities/primitives.h"
#include "../../utilities/sp_type_traits.h"
#include "../../numeric/geometric_algorithm.h"
#include "../../data_structure/data_structure.h"

#if !NO_MPI || USE_MPI
#include "../../parallel/mpi_comm.h"
#endif

namespace simpla
{

/**
 * \ingroup Topology
 */
/**
 *  \brief  structured mesh, n-dimensional array
 */

struct StructuredMesh
{

	typedef StructuredMesh this_type;

	static constexpr size_t ndims = 3;

	typedef unsigned long index_type;
	typedef nTuple<index_type, ndims> index_tuple;

	typedef unsigned long id_type;
	typedef nTuple<Real, ndims> coordinates_type;

	enum
	{
		MAX_DEPTH_OF_TREE = 5
	};
	static constexpr size_t MAX_NUM_NEIGHBOUR_ELEMENT = 12;
	static constexpr size_t MAX_NUM_VERTEX_PER_CEL = 8;
	static constexpr size_t DEFAULT_GHOSTS_WIDTH = 3;

private:
	bool is_valid_ = false;
public:

	Properties properties;
	//***************************************************************************************************

	StructuredMesh()
	{
	}
	StructuredMesh(nTuple<size_t, ndims> const &dims)
	{
		dimensions(&dims[0]);
	}
	virtual ~StructuredMesh()
	{
	}

	this_type & operator=(const this_type&) = delete;

	StructuredMesh(const this_type&) = delete;

	void swap(StructuredMesh & rhs) = delete;

	template<typename TDict>
	bool load(TDict const & dict)
	{
		if (dict["Type"].template as<std::string>() != get_type_as_string())
		{
			RUNTIME_ERROR(
					"Illegal topology type! "
							+ dict["Type"].template as<std::string>());
		}
		else if (!dict["Dimensions"].is_table())
		{
			WARNING << ("Configure  error: no 'Dimensions'!") << std::endl;
			return false;
		}

		else
		{
			VERBOSE << "Load topology : Structured  Mesh " << std::endl;

			dimensions(dict["Dimensions"].template as<nTuple<size_t, 3>>());

			return true;
		}
	}

	template<typename OS>
	OS & print(OS &os) const
	{
		os

		<< " Type = \"" << get_type_as_string() << "\", "

		<< " Dimensions =  " << dimensions() << ","

		<< properties << ",";

		return os;
	}
	static std::string get_type_as_string()
	{
		return std::move(name());
	}
	static std::string name()
	{
		return "StructuredMesh";
	}

	/**@defgroup time
	 * @{
	 */
	unsigned long clock_ = 0UL;

	void next_timestep()
	{
		++clock_;
	}
	unsigned long get_clock() const
	{
		return clock_;
	}

	//!   @}

	bool is_valid() const
	{
		return is_valid_;
	}

	//! @defgroup   Data Set shape
	//! @{

	index_tuple dimensions_;

	index_tuple local_outer_begin_, local_outer_end_, local_outer_count_;

	index_tuple local_inner_begin_, local_inner_end_, local_inner_count_;

	index_tuple ghost_width;

	index_tuple hash_strides_;
	//  \verbatim
	//
	//   |----------------|----------------|---------------|--------------|------------|
	//   ^                ^                ^               ^              ^            ^
	//   |                |                |               |              |            |
	//global          local_outer      local_inner    local_inner    local_outer     global
	// _begin          _begin          _begin           _end           _end          _end
	//
	//  \endverbatim



	//! @name Index Dependent
	//! @{
	//!  signed long is 63bit, unsigned long is 64 bit, add a sign bit

	/**
	 *  \note
	 *  \verbatim
	 * 	Thanks my wife Dr. CHEN Xiang Lan, for her advice on bitwise operation
	 * 	    H          m  I           m    J           m K
	 *  |--------|--------------|--------------|-------------|
	 *  |11111111|00000000000000|00000000000000|0000000000000| <= _MH
	 *  |00000000|11111111111111|00000000000000|0000000000000| <= _MI
	 *  |00000000|00000000000000|11111111111111|0000000000000| <= _MJ
	 *  |00000000|00000000000000|00000000000000|1111111111111| <= _MK
	 *
	 *                      I/J/K
	 *  | INDEX_DIGITS------------------------>|
	 *  |  Root------------------->| Leaf ---->|
	 *  |11111111111111111111111111|00000000000| <=_MRI
	 *  |00000000000000000000000001|00000000000| <=_DI
	 *  |00000000000000000000000000|11111111111| <=_MTI
	 *  | Page NO.->| Tree Root  ->|
	 *  |00000000000|11111111111111|11111111111| <=_MASK
	 *  \endverbatim
	 */

	//! @}
	static constexpr size_t FULL_DIGITS = std::numeric_limits<size_t>::digits;

	static constexpr size_t INDEX_DIGITS = (FULL_DIGITS
			- CountBits<FULL_DIGITS>::n) / 3;

//	static constexpr size_t INDEX_MASK = (1UL << INDEX_DIGITS) - 1;
//
//	static constexpr size_t _DI = (1UL
//			<< (INDEX_DIGITS * 2 + MAX_DEPTH_OF_TREE - 1));
//	static constexpr size_t _DJ =
//			(1UL << (INDEX_DIGITS + MAX_DEPTH_OF_TREE - 1));
//	static constexpr size_t _DK = (1UL << (MAX_DEPTH_OF_TREE - 1));
//	static constexpr size_t _DA = _DI | _DJ | _DK;
//
//	static constexpr size_t INDEX_ROOT_MASK = ((1UL
//			<< (INDEX_DIGITS - MAX_DEPTH_OF_TREE)) - 1) << MAX_DEPTH_OF_TREE;
//
//	static constexpr size_t COMPACT_INDEX_ROOT_MASK = INDEX_ROOT_MASK
//			| (INDEX_ROOT_MASK << INDEX_DIGITS)
//			| (INDEX_ROOT_MASK << INDEX_DIGITS * 2);
//
//	static constexpr size_t NO_HEAD_FLAG = ~((~0UL) << (INDEX_DIGITS * 3));




	static constexpr index_type ZERO_INDEX = 1 << INDEX_DIGITS;
	void dimensions(index_type const * d, index_type const * gw = nullptr)
	{

		if (gw != nullptr)
			ghost_width = gw;

		DataSpace sp(ndims, d, &ghost_width[0]);

		std::tie(std::ignore, dimensions_, local_inner_begin_,
				local_inner_count_, std::ignore, std::ignore) = sp.shape();

		local_inner_end_ = local_inner_begin_ + local_inner_count_;

		std::tie(std::ignore, local_outer_count_, local_outer_begin_,
				local_outer_count_, std::ignore, std::ignore) =
				sp.local_space().shape();

		local_outer_begin_ = local_inner_begin_ - local_outer_begin_;

		local_inner_end_ = local_inner_begin_ + local_inner_count_;

		local_outer_begin_ = local_inner_begin_ - local_outer_begin_;

		local_outer_end_ = local_outer_begin_ + local_outer_count_;

		hash_strides_[2] = 1;
		hash_strides_[1] = local_outer_count_[2] * hash_strides_[2];
		hash_strides_[0] = local_outer_count_[1] * hash_strides_[1];

		update();

	}

	index_tuple const & dimensions() const
	{
		return dimensions_;
	}

	std::pair<coordinates_type, coordinates_type> extents() const
	{
		coordinates_type b, e;
		b = 0;

		for (int i = 0; i < ndims; ++i)
		{
			e[i] = dimensions_[i] > 1 ? 1.0 : 0.0;
		}

		return std::move(std::make_pair(b, e));
	}

	bool check_memory_bounds(id_type s) const
	{
		unsigned mtree = MAX_DEPTH_OF_TREE;
		auto idx = id_to_index(s) >> mtree;
		return

		idx[0] >= local_outer_begin_[0]

		&& idx[0] < local_outer_end_[0]

		&& idx[1] >= local_outer_begin_[1]

		&& idx[1] < local_outer_end_[1]

		&& idx[2] >= local_outer_begin_[2]

		&& idx[2] < local_outer_end_[2]

		;

	}

	template<size_t IForm>
	DataSpace dataspace() const
	{

		size_t rank = ndims;
		nTuple<size_t, MAX_NDIMS_OF_ARRAY> g_dims;
		nTuple<size_t, MAX_NDIMS_OF_ARRAY> g_count;
		nTuple<size_t, MAX_NDIMS_OF_ARRAY> g_offset;
		nTuple<size_t, MAX_NDIMS_OF_ARRAY> g_gw;

		g_dims = dimensions_;
		g_offset = local_inner_begin_;
		g_count = local_inner_count_;
		g_gw = ghost_width;

		if (IForm == EDGE || IForm == FACE)
		{
			g_dims[rank] = 3;
			g_offset[rank] = 0;
			g_count[rank] = 3;
			g_gw[rank] = 0;
			++rank;
		}

		return std::move(DataSpace(rank, &g_dims[0], &g_gw[0]));
	}

	/** @}*/

	coordinates_type dx() const
	{
		coordinates_type res;

		for (int i = 0; i < ndims; ++i)
		{
			res[i] =
					dimensions_[i] > 1 ?
							(1.0 / static_cast<Real>(dimensions_[i])) : 0.0;
		}

		return std::move(res);
	}

	bool in_range(id_type s) const
	{
//		index_tuple idx = decompact(s) >> MAX_DEPTH_OF_TREE;
//
//		return true
//				||
//
//				((dimensions_[0] > 1 && idx[0] < dimensions_[0])
//						&& (dimensions_[1] > 1 || idx[1] < dimensions_[1])
//						&& (dimensions_[2] > 1 || idx[2] < dimensions_[2]))

		return in_local_range(s);
	}

	bool in_local_range(id_type s) const
	{
		auto idx = id_to_index(s) >> MAX_DEPTH_OF_TREE;

		return

		((dimensions_[0] > 1
				|| (idx[0] >= local_inner_begin_[0]
						&& idx[0] < local_inner_end_[0])))

				&& ((dimensions_[1] > 1
						|| (idx[1] >= local_inner_begin_[1]
								&& idx[1] < local_inner_end_[1])))

				&& ((dimensions_[2] > 1
						|| (idx[2] >= local_inner_begin_[2]
								&& idx[2] < local_inner_end_[2])));
	}
	//! @}

	//mask of direction
//	static index_type compact(nTuple<NDIMS, index_type> const & idx )
//	{
//		return
//
//		( static_cast<index_type>( idx[0] & INDEX_MASK) << (INDEX_DIGITS * 2)) |
//
//		( static_cast<index_type>( idx[1] & INDEX_MASK) << (INDEX_DIGITS )) |
//
//		( static_cast<index_type>( idx[2] & INDEX_MASK) )
//
//		;
//	}
	template<typename TS>
	static id_type index_to_id(TS const & x)
	{
		return

		((static_cast<id_type>(x[0] + ZERO_INDEX) & INDEX_MASK)
				<< (INDEX_DIGITS * 2))

				| ((static_cast<id_type>(x[1] + ZERO_INDEX) & INDEX_MASK)
						<< (INDEX_DIGITS))

				| ((static_cast<id_type>(x[2] + ZERO_INDEX) & INDEX_MASK))

		;
	}

	static index_tuple id_to_index(id_type s)
	{

		return std::move(
				index_tuple(
						{ static_cast<index_type>((s >> (INDEX_DIGITS * 2))
								& INDEX_MASK) - ZERO_INDEX,

						static_cast<index_type>((s >> (INDEX_DIGITS))
								& INDEX_MASK) - ZERO_INDEX,

						static_cast<index_type>(s & INDEX_MASK) - ZERO_INDEX

						}));
	}

	template<size_t IFORM>
	static id_type compact_cell_index(index_tuple const & idx, id_type shift)
	{
		index_type mtree = MAX_DEPTH_OF_TREE;
		return index_to_id(idx << mtree) | shift;
	}

	static index_tuple decompact_cell_index(id_type s)
	{
		index_type mtree = MAX_DEPTH_OF_TREE;

		return std::move(id_to_index(s) >> (mtree));
	}
	//! @name Geometry
	//! @{
	Real volume_[8] = { 1, // 000
			1, //001
			1, //010
			1, //011
			1, //100
			1, //101
			1, //110
			1  //111
			};
	Real inv_volume_[8] = { 1, 1, 1, 1, 1, 1, 1, 1 };

	Real dual_volume_[8] = { 1, 1, 1, 1, 1, 1, 1, 1 };

	Real inv_dual_volume_[8] = { 1, 1, 1, 1, 1, 1, 1, 1 };

	nTuple<Real, ndims> inv_extents_, extents_, dx_, inv_dx_;

	bool update()
	{

		for (int i = 0; i < ndims; ++i)
		{
			Real L = static_cast<Real>(dimensions_[i]);
			if (dimensions_[i] <= 1)
			{
				extents_[i] = 0.0;
				inv_extents_[i] = 0.0;
				dx_[i] = 1.0;
				inv_dx_[i] = 1.0;

			}
			else
			{
				extents_[i] = static_cast<Real>((dimensions_[i])
						<< MAX_DEPTH_OF_TREE);
				inv_extents_[i] = 1.0 / extents_[i];

				inv_dx_[i] = L;
				dx_[i] = 1.0 / inv_dx_[i];

			}

			volume_[1UL << (ndims - i - 1)] = dx_[i];
			dual_volume_[7 - (1UL << (ndims - i - 1))] = dx_[i];
			inv_volume_[1UL << (ndims - i - 1)] = inv_dx_[i];
			inv_dual_volume_[7 - (1UL << (ndims - i - 1))] = inv_dx_[i];

		}

		/**
		 * \note
		 *  \verbatim
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
		 * \endverbatim
		 */

		volume_[0] = 1;
//		volume_[1] /* 001 */= dx_[0];
//		volume_[2] /* 010 */= dx_[1];
//		volume_[4] /* 100 */= dx_[2];

		volume_[3] /* 011 */= volume_[1] * volume_[2];
		volume_[5] /* 101 */= volume_[4] * volume_[1];
		volume_[6] /* 110 */= volume_[2] * volume_[4];

		volume_[7] /* 111 */= volume_[1] * volume_[2] * volume_[4];

		dual_volume_[7] = 1;
//		dual_volume_[6] /* 001 */= dx_[0];
//		dual_volume_[5] /* 010 */= dx_[1];
//		dual_volume_[3] /* 100 */= dx_[2];

		dual_volume_[4] /* 011 */= dual_volume_[6] * dual_volume_[5];
		dual_volume_[2] /* 101 */= dual_volume_[3] * dual_volume_[6];
		dual_volume_[1] /* 110 */= dual_volume_[5] * dual_volume_[3];

		dual_volume_[0] /* 111 */= dual_volume_[6] * dual_volume_[5]
				* dual_volume_[3];

		inv_volume_[0] = 1;
//		inv_volume_[1] /* 001 */= inv_dx_[0];
//		inv_volume_[2] /* 010 */= inv_dx_[1];
//		inv_volume_[4] /* 100 */= inv_dx_[2];

		inv_volume_[3] /* 011 */= inv_volume_[1] * inv_volume_[2];
		inv_volume_[5] /* 101 */= inv_volume_[4] * inv_volume_[1];
		inv_volume_[6] /* 110 */= inv_volume_[2] * inv_volume_[4];

		inv_volume_[7] /* 111 */= inv_volume_[1] * inv_volume_[2]
				* inv_volume_[4];

		inv_dual_volume_[7] = 1;
//		inv_dual_volume_[6] /* 001 */= inv_dx_[0];
//		inv_dual_volume_[5] /* 010 */= inv_dx_[1];
//		inv_dual_volume_[3] /* 100 */= inv_dx_[2];

		inv_dual_volume_[4] /* 011 */= inv_dual_volume_[6]
				* inv_dual_volume_[5];
		inv_dual_volume_[2] /* 101 */= inv_dual_volume_[3]
				* inv_dual_volume_[6];
		inv_dual_volume_[1] /* 110 */= inv_dual_volume_[5]
				* inv_dual_volume_[3];

		inv_dual_volume_[0] /* 111 */= inv_dual_volume_[6] * inv_dual_volume_[5]
				* inv_dual_volume_[3];

		is_valid_ = true;

		return is_valid_;
	}

	Real const & volume(id_type s) const
	{
		return volume_[node_id(s)];
	}

	Real inv_volume(id_type s) const
	{
		return inv_volume_[node_id(s)];
	}

	Real dual_volume(id_type s) const
	{
		return dual_volume_[node_id(s)];
	}

	Real inv_dual_volume(id_type s) const
	{
		return inv_dual_volume_[node_id(s)];
	}

	Real cell_volume(id_type s) const
	{
		return volume_[1] * volume_[2] * volume_[4];
	}

	Real volume(id_type s, std::integral_constant<bool, false>) const
	{
		return volume(s);
	}

	Real inv_volume(id_type s, std::integral_constant<bool, false>) const
	{
		return inv_volume(s);
	}

	Real inv_dual_volume(id_type s, std::integral_constant<bool, false>) const
	{
		return inv_dual_volume(s);
	}
	Real dual_volume(id_type s, std::integral_constant<bool, false>) const
	{
		return dual_volume(s);
	}

	Real volume(id_type s, std::integral_constant<bool, true>) const
	{
		return in_range(s) ? volume(s) : 0.0;
	}

	Real inv_volume(id_type s, std::integral_constant<bool, true>) const
	{
		return in_range(s) ? inv_volume(s) : 0.0;
	}

	Real dual_volume(id_type s, std::integral_constant<bool, true>) const
	{
		return in_range(s) ? dual_volume(s) : 0.0;
	}
	Real inv_dual_volume(id_type s, std::integral_constant<bool, true>) const
	{
		return in_range(s) ? inv_dual_volume(s) : 0.0;
	}

	//! @}

	//! @name Coordinates
	//! @{

	/***
	 *
	 * @param s
	 * @return Coordinates range in [0,1)
	 */

	inline coordinates_type index_to_coordinates(index_tuple const&idx) const
	{

		return std::move(coordinates_type( {

		static_cast<Real>(idx[0]) * inv_extents_[0],

		static_cast<Real>(idx[1]) * inv_extents_[1],

		static_cast<Real>(idx[2]) * inv_extents_[2] }));
	}

	inline index_tuple coordinates_to_index(coordinates_type x) const
	{
		return std::move(index_tuple( {

		static_cast<index_type>(x[0] * extents_[0]),

		static_cast<index_type>(x[1] * extents_[1]),

		static_cast<index_type>(x[2] * extents_[2])

		}));
	}

	inline index_tuple to_cell_index(index_tuple idx) const
	{
		idx = idx >> MAX_DEPTH_OF_TREE;

		return std::move(index_tuple(idx));
	}

	inline coordinates_type id_to_coordinates(id_type s) const
	{
		return std::move(index_to_coordinates(id_to_index(s)));
	}

	inline coordinates_type coordinates_local_to_global(id_type s,
			coordinates_type r) const
	{
		Real CELL_SCALE_R = static_cast<Real>(1UL << (MAX_DEPTH_OF_TREE));
		Real INV_CELL_SCALE_R = 1.0 / CELL_SCALE_R;

		coordinates_type x;

		x = r + ((id_to_index(s) >> MAX_DEPTH_OF_TREE))
				+ 0.5 * (id_to_index((s & _DA)) >> (MAX_DEPTH_OF_TREE - 1));

		x[0] *= dx_[0];
		x[1] *= dx_[1];
		x[2] *= dx_[2];

		return std::move(x);
	}

	template<typename TI>
	inline auto coordinates_local_to_global(TI const& idx) const
	DECL_RET_TYPE (coordinates_local_to_global(std::get<0> (idx),
					std::get<1> (idx)))

	/**
	 *
	 * @param x coordinates \f$ x \in \left[0,1\right)\f$
	 * @param shift
	 * @return s,r  s is the largest grid point not greater than x.
	 *       and  \f$ r \in \left[0,1.0\right) \f$ is the normalize  distance between x and s
	 */
	inline std::tuple<id_type, coordinates_type> coordinates_global_to_local(
			coordinates_type const & x, id_type shift = 0UL) const
	{

		index_tuple I = id_to_index(shift >> (MAX_DEPTH_OF_TREE - 1));

		coordinates_type r;

		r[0] = x[0] * dimensions_[0] - 0.5 * static_cast<Real>(I[0]);
		r[1] = x[1] * dimensions_[1] - 0.5 * static_cast<Real>(I[1]);
		r[2] = x[2] * dimensions_[2] - 0.5 * static_cast<Real>(I[2]);

		I[0] = static_cast<index_type>(std::floor(r[0]));
		I[1] = static_cast<index_type>(std::floor(r[1]));
		I[2] = static_cast<index_type>(std::floor(r[2]));

		r -= I;

		id_type s = (index_to_id(I) << MAX_DEPTH_OF_TREE) | shift;

		return std::move(std::make_tuple(s, r));
	}

	/**
	 *
	 * @param x  coordinates \f$ x \in \left[0,1\right)\f$
	 * @param shift
	 * @return s,r   s is thte conmpact index of nearest grid point
	 *    and  \f$ r \in \left[-0.5,0.5\right) \f$   is the normalize  distance between x and s
	 */
	inline std::tuple<id_type, coordinates_type> coordinates_global_to_local_NGP(
			std::tuple<id_type, coordinates_type> const &z) const
	{
		auto & x = std::get<1>(z);
		id_type shift = std::get<0>(z);

		index_tuple I = id_to_index(shift >> (MAX_DEPTH_OF_TREE - 1));

		coordinates_type r;

		r[0] = x[0] * dimensions_[0] - 0.5 * static_cast<Real>(I[0]);
		r[1] = x[1] * dimensions_[1] - 0.5 * static_cast<Real>(I[1]);
		r[2] = x[2] * dimensions_[2] - 0.5 * static_cast<Real>(I[2]);

		I[0] = static_cast<index_type>(std::floor(r[0] + 0.5));
		I[1] = static_cast<index_type>(std::floor(r[1] + 0.5));
		I[2] = static_cast<index_type>(std::floor(r[2] + 0.5));

		r -= I;

		id_type s = (index_to_id(I) << MAX_DEPTH_OF_TREE) | shift;

		return std::move(std::make_tuple(s, r));
	}

	//! @}

	//! @name Index auxiliary functions
	//! @{
	static id_type dual(id_type r)
	{
		return (r & (~_DA)) | ((~(r & _DA)) & _DA);

	}
	static id_type get_cell_index(id_type r)
	{
//		index_type mask = (1UL << (INDEX_DIGITS - DepthOfTree(r))) - 1;
//
//		return r & (~(mask | (mask << INDEX_DIGITS) | (mask << (INDEX_DIGITS * 2))));
		return r & COMPACT_INDEX_ROOT_MASK;
	}
	static id_type node_id(id_type s)
	{

		return (((s >> (INDEX_DIGITS * 2 + MAX_DEPTH_OF_TREE - 1)) & 1UL) << 2)
				| (((s >> (INDEX_DIGITS + MAX_DEPTH_OF_TREE - 1)) & 1UL) << 1)
				| ((s >> (MAX_DEPTH_OF_TREE - 1)) & 1UL);

	}

	id_type get_shift(id_type nodeid, id_type h = 0UL) const
	{

		return

		(((nodeid & 4UL) >> 2) << (INDEX_DIGITS * 2 + MAX_DEPTH_OF_TREE - 1)) |

		(((nodeid & 2UL) >> 1) << (INDEX_DIGITS + MAX_DEPTH_OF_TREE - 1)) |

		((nodeid & 1UL) << (MAX_DEPTH_OF_TREE - 1));

	}

	id_type get_first_node_shift(id_type iform) const
	{
		id_type nid;
		switch (iform)
		{
		case VERTEX:
			nid = 0;
			break;
		case EDGE:
			nid = 4;
			break;
		case FACE:
			nid = 3;
			break;
		case VOLUME:
			nid = 7;
			break;
		}

		return get_shift(nid);
	}

	static size_t get_num_of_comp_per_cell(size_t iform)
	{
		size_t res;
		switch (iform)
		{
		case VERTEX:
			res = 1;
			break;
		case EDGE:
			res = 3;
			break;
		case FACE:
			res = 3;
			break;
		case VOLUME:
			res = 1;
			break;
		}

		return res;
	}

	static id_type roate(id_type r)
	{

		return (r & (~_DA))

		| ((r & (((_DI | _DJ)))) >> INDEX_DIGITS)

		| ((r & (((_DK)))) << (INDEX_DIGITS * 2));

	}

	/**
	 *  rotate vector direction  mask
	 *  (1/2,0,0) => (0,0,1/2) or   (1/2,1/2,0) => (1/2,0,1/2)
	 * @param s
	 * @return
	 */

	static id_type inverse_roate(id_type s)
	{

		return (s & (~(_DA)))

		| ((s & (((_DK | _DJ)))) << INDEX_DIGITS)

		| ((s & (((_DI)))) >> (INDEX_DIGITS * 2));

	}

	static id_type delta_index(id_type r)
	{
		return (r & _DA);
	}

	static id_type DI(size_t i, id_type r)
	{
		return (1UL << (INDEX_DIGITS * (ndims - i - 1) + MAX_DEPTH_OF_TREE - 1));

	}
	static id_type delta_index(size_t i, id_type r)
	{
		return DI(i, r) & r;
	}

	/**
	 * Get component number or vector direction
	 * @param s
	 * @return
	 */
	static id_type component_number(id_type s)
	{
		id_type res = 0;
		switch (node_id(s))
		{
		case 4:
		case 3:
			res = 0;
			break;
		case 2:
		case 5:
			res = 1;
			break;
		case 1:
		case 6:
			res = 2;
			break;
		}
		return res;
	}

	static id_type IForm(id_type r)
	{
		id_type res = 0;
		switch (node_id(r))
		{
		case 0:
			res = VERTEX;
			break;
		case 1:
		case 2:
		case 4:
			res = EDGE;
			break;

		case 3:
		case 5:
		case 6:
			res = FACE;
			break;

		case 7:
			res = VOLUME;
		}
		return res;
	}
	//! @}

	template<size_t IForm> struct Range;

	/**
	 *  @name Select
	 *  @{
	 */
private:
	template<size_t IFORM>
	Range<IFORM> select_rectangle_(index_tuple const &ib, index_tuple const &ie,
			index_tuple b, index_tuple e) const
	{
		clipping(ib, ie, &b, &e);

		return std::move(Range<IFORM>(b, e));

	}
public:

	template<size_t IFORM>
	Range<IFORM> select() const
	{
		return (Range<IFORM>(local_inner_begin_, local_inner_end_));
	}

	/**
	 * \fn Select
	 * \brief
	 * @param range
	 * @param b
	 * @param e
	 * @return
	 */
	template<size_t IForm>
	auto select(index_tuple const & b, index_tuple const &e) const
	DECL_RET_TYPE(select_rectangle_<IForm>( b, e,
					local_inner_begin_, local_inner_end_))

	/**
	 *
	 */
	template<size_t IFORM>
	Range<IFORM> select(coordinates_type const & xmin,
			coordinates_type const &xmax) const
	{
		return std::move(Range<IFORM>());
	}

	template<size_t IFORM>
	Range<IFORM> select_outer() const
	{
		return std::move(Range<IFORM>(local_outer_begin_, local_outer_end_));
	}

	/**
	 * \fn Select
	 * \brief
	 * @param range
	 * @param b
	 * @param e
	 * @return
	 */
	template<size_t IFORM>
	auto select_outer(index_tuple const & b, index_tuple const &e) const
	DECL_RET_TYPE (select_rectangle_<IFORM>( b, e, local_outer_begin_,
					local_outer_end_))

	template<size_t IFORM>
	auto select_inner(index_tuple const & b, index_tuple const & e) const
	DECL_RET_TYPE (select_rectangle_<IFORM>( b, e, local_inner_begin_,
					local_inner_end_))

	/**  @} */
	/**
	 *  @name Hash
	 *  @{
	 */

	template<size_t IFORM>
	size_t max_hash() const
	{
		return NProduct(local_outer_count_)
				* ((IFORM == EDGE || IFORM == FACE) ? 3 : 1);
	}

	static index_type mod_(index_type a, index_type L)
	{
		return (a + L) % L;
	}

	size_t hash(id_type s) const
	{
		id_type m_tree = MAX_DEPTH_OF_TREE;
		nTuple<index_type, ndims> d = id_to_index(s) - local_outer_begin_;

		size_t res =

		mod_(d[0], (local_outer_count_[0])) * hash_strides_[0] +

		mod_(d[1], (local_outer_count_[1])) * hash_strides_[1] +

		mod_(d[2], (local_outer_count_[2])) * hash_strides_[2];

		switch (node_id(s))
		{
		case 4:
		case 3:
			res = ((res << 1) + res);
			break;
		case 2:
		case 5:
			res = ((res << 1) + res) + 1;
			break;
		case 1:
		case 6:
			res = ((res << 1) + res) + 2;
			break;
		}

		return res;

	}

	/** @}*/

	/** @name   Topology
	 *  @{
	 */

	inline size_t get_vertices(id_type s, id_type *v) const
	{
		size_t n = 0;
		switch (IForm(s))
		{
		case VERTEX:
		{
			v[0] = s;
		}
			n = 1;
			break;
		case EDGE:
		{
			auto di = delta_index(s);
			v[0] = s + di;
			v[1] = s - di;
		}
			n = 2;
			break;

		case FACE:
		{
			auto di = delta_index(roate(dual(s)));
			auto dj = delta_index(inverse_roate(dual(s)));

			v[0] = s - di - dj;
			v[1] = s - di - dj;
			v[2] = s + di + dj;
			v[3] = s + di + dj;
			n = 4;
		}
			break;
		case VOLUME:
		{
			auto di = DI(0, s);
			auto dj = DI(1, s);
			auto dk = DI(2, s);

			v[0] = ((s - di) - dj) - dk;
			v[1] = ((s - di) - dj) + dk;
			v[2] = ((s - di) + dj) - dk;
			v[3] = ((s - di) + dj) + dk;

			v[4] = ((s + di) - dj) - dk;
			v[5] = ((s + di) - dj) + dk;
			v[6] = ((s + di) + dj) - dk;
			v[7] = ((s + di) + dj) + dk;
			n = 8;
		}
			break;
		}
		return n;
	}

	template<size_t I>
	inline size_t get_adjacent_cells(std::integral_constant<size_t, I>,
			std::integral_constant<size_t, I>, id_type s, id_type *v) const
	{
		v[0] = s;
		return 1;
	}

	inline size_t get_adjacent_cells(std::integral_constant<size_t, EDGE>,
			std::integral_constant<size_t, VERTEX>, id_type s, id_type *v) const
	{
		v[0] = s + delta_index(s);
		v[1] = s - delta_index(s);
		return 2;
	}

	inline size_t get_adjacent_cells(std::integral_constant<size_t, FACE>,
			std::integral_constant<size_t, VERTEX>, id_type s, id_type *v) const
	{
		/**
		 * \verbatim
		 *                ^y
		 *               /
		 *        z     /
		 *        ^
		 *        |   2---------------*
		 *        |  /|              /|
		 *          / |             / |
		 *         /  |            /  |
		 *        3---|-----------*   |
		 *        | m |           |   |
		 *        |   1-----------|---*
		 *        |  /            |  /
		 *        | /             | /
		 *        |/              |/
		 *        0---------------*---> x
		 * \endverbatim
		 *
		 */

		auto di = delta_index(roate(dual(s)));
		auto dj = delta_index(inverse_roate(dual(s)));

		v[0] = s - di - dj;
		v[1] = s - di - dj;
		v[2] = s + di + dj;
		v[3] = s + di + dj;

		return 4;
	}

	inline size_t get_adjacent_cells(std::integral_constant<size_t, VOLUME>,
			std::integral_constant<size_t, VERTEX>, id_type s, id_type *v) const
	{
		/**
		 * \verbatim
		 *                ^y
		 *               /
		 *        z     /
		 *        ^
		 *        |   6---------------7
		 *        |  /|              /|
		 *          / |             / |
		 *         /  |            /  |
		 *        4---|-----------5   |
		 *        |   |           |   |
		 *        |   2-----------|---3
		 *        |  /            |  /
		 *        | /             | /
		 *        |/              |/
		 *        0---------------1   ---> x
		 *
		 *   \endverbatim
		 */
		auto di = DI(0, s);
		auto dj = DI(1, s);
		auto dk = DI(2, s);

		v[0] = ((s - di) - dj) - dk;
		v[1] = ((s - di) - dj) + dk;
		v[2] = ((s - di) + dj) - dk;
		v[3] = ((s - di) + dj) + dk;

		v[4] = ((s + di) - dj) - dk;
		v[5] = ((s + di) - dj) + dk;
		v[6] = ((s + di) + dj) - dk;
		v[7] = ((s + di) + dj) + dk;

		return 8;
	}

	inline size_t get_adjacent_cells(std::integral_constant<size_t, VERTEX>,
			std::integral_constant<size_t, EDGE>, id_type s, id_type *v) const
	{
		/**
		 * \verbatim
		 *
		 *                ^y
		 *               /
		 *        z     /
		 *        ^
		 *        |   6---------------7
		 *        |  /|              /|
		 *          2 |             / |
		 *         /  1            /  |
		 *        4---|-----------5   |
		 *        |   |           |   |
		 *        |   2-----------|---3
		 *        3  /            |  /
		 *        | 0             | /
		 *        |/              |/
		 *        0------E0-------1   ---> x
		 *
		 * \endverbatim
		 */

		auto di = DI(0, s);
		auto dj = DI(1, s);
		auto dk = DI(2, s);

		v[0] = s + di;
		v[1] = s - di;

		v[2] = s + dj;
		v[3] = s - dj;

		v[4] = s + dk;
		v[5] = s - dk;

		return 6;
	}

	inline size_t get_adjacent_cells(std::integral_constant<size_t, FACE>,
			std::integral_constant<size_t, EDGE>, id_type s, id_type *v) const
	{

		/**
		 *\verbatim
		 *                ^y
		 *               /
		 *        z     /
		 *        ^
		 *        |   6---------------7
		 *        |  /|              /|
		 *          2 |             / |
		 *         /  1            /  |
		 *        4---|-----------5   |
		 *        |   |           |   |
		 *        |   2-----------|---3
		 *        3  /            |  /
		 *        | 0             | /
		 *        |/              |/
		 *        0---------------1   ---> x
		 *
		 *\endverbatim
		 */
		auto d1 = delta_index(roate(dual(s)));
		auto d2 = delta_index(inverse_roate(dual(s)));
		v[0] = s - d1;
		v[1] = s + d1;
		v[2] = s - d2;
		v[3] = s + d2;

		return 4;
	}

	inline size_t get_adjacent_cells(std::integral_constant<size_t, VOLUME>,
			std::integral_constant<size_t, EDGE>, id_type s, id_type *v) const
	{

		/**
		 *\verbatim
		 *                ^y
		 *               /
		 *        z     /
		 *        ^
		 *        |   6------10-------7
		 *        |  /|              /|
		 *         11 |             9 |
		 *         /  7            /  6
		 *        4---|---8-------5   |
		 *        |   |           |   |
		 *        |   2-------2---|---3
		 *        4  /            5  /
		 *        | 3             | 1
		 *        |/              |/
		 *        0-------0-------1   ---> x
		 *
		 *\endverbatim
		 */
		auto di = DI(0, s);
		auto dj = DI(1, s);
		auto dk = DI(2, s);

		v[0] = (s + di) + dj;
		v[1] = (s + di) - dj;
		v[2] = (s - di) + dj;
		v[3] = (s - di) - dj;

		v[4] = (s + dk) + dj;
		v[5] = (s + dk) - dj;
		v[6] = (s - dk) + dj;
		v[7] = (s - dk) - dj;

		v[8] = (s + di) + dk;
		v[9] = (s + di) - dk;
		v[10] = (s - di) + dk;
		v[11] = (s - di) - dk;

		return 12;
	}

	inline size_t get_adjacent_cells(std::integral_constant<size_t, VERTEX>,
			std::integral_constant<size_t, FACE>, id_type s, id_type *v) const
	{
		/**
		 *\verbatim
		 *                ^y
		 *               /
		 *        z     /
		 *        ^
		 *        |   6---------------7
		 *        |  /|              /|
		 *        | / |             / |
		 *        |/  |            /  |
		 *        4---|-----------5   |
		 *        |   |           |   |
		 *        | 0 2-----------|---3
		 *        |  /            |  /
		 *   11   | /      8      | /
		 *      3 |/              |/
		 * -------0---------------1   ---> x
		 *       /| 1
		 *10    / |     9
		 *     /  |
		 *      2 |
		 *
		 *
		 *
		 *              |
		 *          7   |   4
		 *              |
		 *      --------*---------
		 *              |
		 *          6   |   5
		 *              |
		 *
		 *\endverbatim
		 */
		auto di = DI(0, s);
		auto dj = DI(1, s);
		auto dk = DI(2, s);

		v[0] = (s + di) + dj;
		v[1] = (s + di) - dj;
		v[2] = (s - di) + dj;
		v[3] = (s - di) - dj;

		v[4] = (s + dk) + dj;
		v[5] = (s + dk) - dj;
		v[6] = (s - dk) + dj;
		v[7] = (s - dk) - dj;

		v[8] = (s + di) + dk;
		v[9] = (s + di) - dk;
		v[10] = (s - di) + dk;
		v[11] = (s - di) - dk;

		return 12;
	}

	inline size_t get_adjacent_cells(std::integral_constant<size_t, EDGE>,
			std::integral_constant<size_t, FACE>, id_type s, id_type *v) const
	{

		/**
		 *\verbatim
		 *                ^y
		 *               /
		 *        z     /
		 *        ^
		 *        |   6---------------7
		 *        |  /|              /|
		 *        | / |             / |
		 *        |/  |            /  |
		 *        4---|-----------5   |
		 *        |   |           |   |
		 *        |   2-----------|---3
		 *        |  /  0         |  /
		 *        | /      1      | /
		 *        |/              |/
		 * -------0---------------1   ---> x
		 *       /|
		 *      / |   3
		 *     /  |       2
		 *        |
		 *
		 *
		 *
		 *              |
		 *          7   |   4
		 *              |
		 *      --------*---------
		 *              |
		 *          6   |   5
		 *              |
		 *
		 *\endverbatim
		 */

		auto d1 = delta_index(roate((s)));
		auto d2 = delta_index(inverse_roate((s)));

		v[0] = s - d1;
		v[1] = s + d1;
		v[2] = s - d2;
		v[3] = s + d2;

		return 4;
	}

	inline size_t get_adjacent_cells(std::integral_constant<size_t, VOLUME>,
			std::integral_constant<size_t, FACE>, id_type s, id_type *v) const
	{

		/**
		 *\verbatim
		 *                ^y
		 *               /
		 *        z     /
		 *        ^    /
		 *        |   6---------------7
		 *        |  /|              /|
		 *        | / |    5        / |
		 *        |/  |     1      /  |
		 *        4---|-----------5   |
		 *        | 0 |           | 2 |
		 *        |   2-----------|---3
		 *        |  /    3       |  /
		 *        | /       4     | /
		 *        |/              |/
		 * -------0---------------1   ---> x
		 *       /|
		 *\endverbatim
		 */

		auto di = DI(0, s);
		auto dj = DI(1, s);
		auto dk = DI(2, s);

		v[0] = s - di;
		v[1] = s + di;

		v[2] = s - di;
		v[3] = s + dj;

		v[4] = s - dk;
		v[5] = s + dk;

		return 6;
	}

	inline size_t get_adjacent_cells(std::integral_constant<size_t, VERTEX>,
			std::integral_constant<size_t, VOLUME>, id_type s, id_type *v) const
	{
		/**
		 *\verbatim
		 *                ^y
		 *               /
		 *        z     /
		 *        ^
		 *        |   6---------------7
		 *        |  /|              /|
		 *        | / |             / |
		 *        |/  |            /  |
		 *        4---|-----------5   |
		 *   3    |   |    0      |   |
		 *        |   2-----------|---3
		 *        |  /            |  /
		 *        | /             | /
		 *        |/              |/
		 * -------0---------------1   ---> x
		 *  3    /|       1
		 *      / |
		 *     /  |
		 *        |
		 *
		 *
		 *
		 *              |
		 *          7   |   4
		 *              |
		 *      --------*---------
		 *              |
		 *          6   |   5
		 *              |
		 *
		 *\endverbatim
		 */

		auto di = DI(0, s);
		auto dj = DI(1, s);
		auto dk = DI(2, s);

		v[0] = ((s - di) - dj) - dk;
		v[1] = ((s - di) - dj) + dk;
		v[2] = ((s - di) + dj) - dk;
		v[3] = ((s - di) + dj) + dk;

		v[4] = ((s + di) - dj) - dk;
		v[5] = ((s + di) - dj) + dk;
		v[6] = ((s + di) + dj) - dk;
		v[7] = ((s + di) + dj) + dk;

		return 8;
	}

	inline size_t get_adjacent_cells(std::integral_constant<size_t, EDGE>,
			std::integral_constant<size_t, VOLUME>, id_type s, id_type *v) const
	{

		/**
		 *\verbatim
		 *                ^y
		 *               /
		 *        z     /
		 *        ^
		 *        |   6---------------7
		 *        |  /|              /|
		 *        | / |             / |
		 *        |/  |            /  |
		 *        4---|-----------5   |
		 *        |   |           |   |
		 *        |   2-----------|---3
		 *        |  /  0         |  /
		 *        | /      1      | /
		 *        |/              |/
		 * -------0---------------1   ---> x
		 *       /|
		 *      / |   3
		 *     /  |       2
		 *        |
		 *
		 *
		 *
		 *              |
		 *          7   |   4
		 *              |
		 *      --------*---------
		 *              |
		 *          6   |   5
		 *              |
		 *
		 *\endverbatim
		 */

		auto d1 = delta_index(roate((s)));
		auto d2 = delta_index(inverse_roate((s)));

		v[0] = s - d1 - d2;
		v[1] = s + d1 - d2;
		v[2] = s - d1 + d2;
		v[3] = s + d1 + d2;
		return 4;
	}

	inline size_t get_adjacent_cells(std::integral_constant<size_t, FACE>,
			std::integral_constant<size_t, VOLUME>, id_type s, id_type *v) const
	{

		/**
		 *\verbatim
		 *                ^y
		 *               /
		 *        z     /
		 *        ^    /
		 *        |   6---------------7
		 *        |  /|              /|
		 *        | / |             / |
		 *        |/  |            /  |
		 *        4---|-----------5   |
		 *        | 0 |           |   |
		 *        |   2-----------|---3
		 *        |  /            |  /
		 *        | /             | /
		 *        |/              |/
		 * -------0---------------1   ---> x
		 *       /|
		 *\endverbatim
		 */

		auto d = delta_index(dual(s));
		v[0] = s + d;
		v[1] = s - d;

		return 2;
	}
	/** @}*/

	template<typename TV>
	TV sample_(std::integral_constant<size_t, VERTEX>, size_t s,
			TV const &v) const
	{
		return v;
	}

	template<typename TV>
	TV sample_(std::integral_constant<size_t, VOLUME>, size_t s,
			TV const &v) const
	{
		return v;
	}

	template<typename TV>
	TV sample_(std::integral_constant<size_t, EDGE>, size_t s,
			nTuple<TV, 3> const &v) const
	{
		return v[component_number(s)];
	}

	template<typename TV>
	TV sample_(std::integral_constant<size_t, FACE>, size_t s,
			nTuple<TV, 3> const &v) const
	{
		return v[component_number(s)];
	}

	template<size_t IFORM, typename TV>
	TV sample_(std::integral_constant<size_t, IFORM>, size_t s,
			TV const & v) const
	{
		return v;
	}

	template<size_t IFORM, typename ...Args>
	auto sample(Args && ... args) const
	DECL_RET_TYPE((sample_(std::integral_constant<size_t, IFORM>(),
							std::forward<Args>(args)...)))

}
;
//// class UniformArray
//inline StructuredMesh::range_type split(
//		StructuredMesh::range_type const & range, size_t num_process,
//		size_t process_num, size_t ghost_width = 0)
//{
//
//	static constexpr size_t ndims = StructuredMesh::ndims;
//
//	StructuredMesh::iterator ib = begin(range);
//	StructuredMesh::iterator ie = end(range);
//
//	auto b = ib.self_;
//	decltype(b) e = (--ie).self_ + 1;
//
//	auto shift = ib.shift_;
//
//	decltype(b) count;
//	count = e - b;
//
//	int n = 0;
//	size_t L = 0;
//	for (int i = 0; i < ndims; ++i)
//	{
//		if (count[i] > L)
//		{
//			L = count[i];
//			n = i;
//		}
//	}
//
//	if ((2 * ghost_width * num_process > count[n] || num_process > count[n]))
//	{
//		if (process_num > 0)
//			count = 0;
//	}
//	else
//	{
//		e[n] = b[n] + (count[n] * (process_num + 1)) / num_process;
//		b[n] += (count[n] * process_num) / num_process;
//
//	}
//
//	return std::move(StructuredMesh::range_type(b, e, shift));
//}

}
// namespace simpla

//namespace std
//{
//
//typename iterator_traits<simpla::StructuredMesh::iterator>::difference_type inline //
//distance(simpla::StructuredMesh::iterator b, simpla::StructuredMesh::iterator e)
//{
//
//	typename simpla::StructuredMesh::iterator::difference_type res;
//
//	--e;
//
//	res = simpla::NProduct((e).self_ - b.self_ + 1);
//
//	switch (simpla::StructuredMesh::IForm(b.shift_))
//	{
//	case simpla::EDGE:
//	case simpla::FACE:
//		res = res * 3
//		        + (simpla::StructuredMesh::component_number(e.shift_)
//		                - simpla::StructuredMesh::component_number(b.shift_)) + 1;
//		break;
//	}
//
//	return res;
//}
//}  // namespace std

#endif /* STRUCTURED_H_ */
