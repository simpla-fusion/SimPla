/**
 * @file  structured.h
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

#include "../../data_representation/data_interface.h"
#include "../../utilities/utilities.h"
#include "../../gtl/ntuple.h"
#include "../../gtl/primitives.h"
#include "../../numeric/geometric_algorithm.h"
#include "../diff_geometry_common.h"

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
struct StructuredMesh
{

	typedef StructuredMesh this_type;

	static constexpr size_t ndims = 3;

	typedef unsigned long index_type;
	typedef nTuple<index_type, ndims> index_tuple;

	typedef unsigned long id_type;
	typedef nTuple<Real, ndims> coordinates_type;

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
	template<typename ...Args>
	StructuredMesh(nTuple<size_t, ndims> const &count,
			nTuple<size_t, ndims> const &offset)
	{
		dimensions(count, offset);
	}

	~StructuredMesh() = default;

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
		return "StructuredMesh";
	}

	bool is_valid() const
	{
		return is_valid_;
	}

	/**
	 * @name  Data Shape
	 * @{
	 **/

	index_tuple count_, offset_;

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

	/**
	 *  signed long is 63bit, unsigned long is 64 bit, add a sign bit
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

	static constexpr size_t FULL_DIGITS = std::numeric_limits<size_t>::digits;

	static constexpr size_t INDEX_DIGITS = (FULL_DIGITS
			- CountBits<FULL_DIGITS>::n) / 3;

	static constexpr size_t FLOATING_POINT_POS = 4;

	static constexpr index_type FLOATING_POINT_FACTOR = 1 << FLOATING_POINT_POS;

	static constexpr size_t INDEX_MASK = (1UL << (INDEX_DIGITS)) - 1;

	static constexpr size_t D_INDEX = (1UL << (FLOATING_POINT_POS));

	static constexpr size_t _DK = D_INDEX << (INDEX_DIGITS * 2 - 1);

	static constexpr size_t _DJ = D_INDEX << (INDEX_DIGITS - 1);

	static constexpr size_t _DI = D_INDEX >> 1;

	static constexpr size_t _DA = _DI | _DJ | _DK;

	static constexpr size_t CELL_ID_MASK_ = ((1UL
			<< (INDEX_DIGITS - FLOATING_POINT_POS)) - 1) << FLOATING_POINT_POS;

	static constexpr size_t CELL_ID_MASK =

	(CELL_ID_MASK_ << (INDEX_DIGITS * 2))

	| (CELL_ID_MASK_ << (INDEX_DIGITS))

	| (CELL_ID_MASK_);

//	static constexpr size_t COMPACT_INDEX_ROOT_MASK = INDEX_ROOT_MASK
//			| (INDEX_ROOT_MASK << INDEX_DIGITS)
//			| (INDEX_ROOT_MASK << INDEX_DIGITS * 2);
//
//	static constexpr size_t NO_HEAD_FLAG = ~((~0UL) << (INDEX_DIGITS * 3));

	static constexpr index_type ZERO_INDEX = 1 << (INDEX_DIGITS - 1);
	void dimensions(index_tuple const & count)
	{
		index_tuple offset;
		offset = 0;
		dimensions(count, offset);
	}
	void dimensions(index_tuple const & count, index_tuple const & offset,
			index_type const * gw = nullptr)
	{

		count_ = count;
		offset_ = offset;

		local_outer_count_ = count_ << FLOATING_POINT_POS;
		local_outer_begin_ = ZERO_INDEX;
		local_outer_end_ = local_outer_begin_ + local_outer_count_;

		local_inner_count_ = count_ << FLOATING_POINT_POS;
		local_inner_begin_ = ZERO_INDEX;
		local_inner_end_ = local_inner_begin_ + local_inner_count_;
//		if (gw != nullptr)
//			ghost_width = gw;
//
//		DataSpace sp(ndims, d, &ghost_width[0]);
//
//		std::tie(std::ignore, dimensions_, local_inner_begin_,
//				local_inner_count_, std::ignore, std::ignore) = sp.shape();
//
//		local_inner_end_ = local_inner_begin_ + local_inner_count_;
//
//		std::tie(std::ignore, local_outer_count_, local_outer_begin_,
//				local_outer_count_, std::ignore, std::ignore) =
//				sp.local_space().shape();
//
//		local_outer_begin_ = local_inner_begin_ - local_outer_begin_;
//
//		local_inner_end_ = local_inner_begin_ + local_inner_count_;
//
//		local_outer_begin_ = local_inner_begin_ - local_outer_begin_;
//
//		local_outer_end_ = local_outer_begin_ + local_outer_count_;
//
		hash_strides_[2] = 1;
		hash_strides_[1] = count_[2] * hash_strides_[2];
		hash_strides_[0] = count_[1] * hash_strides_[1];

		update();

	}

	index_tuple const & dimensions() const
	{
		return count_;
	}

	bool check_memory_bounds(id_type s) const
	{
		auto idx = id_to_index(s) >> FLOATING_POINT_POS;
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

		g_dims = count_;
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

	template<size_t IFORM>
	static constexpr id_type compact_cell_index(index_tuple const & idx,
			id_type shift)
	{
		return index_to_id(idx << FLOATING_POINT_POS) | shift;
	}

	static index_tuple decompact_cell_index(id_type s)
	{
		return std::move(id_to_index(s) >> (FLOATING_POINT_POS));
	}

	/**
	 *   @name Geometry
	 *   For For uniform structured grid, the volume of cell is 1.0
	 *   and dx=1.0
	 *   @{
	 */

	static constexpr Real COORDINATES_TO_INDEX_FACTOR = static_cast<Real>(1
			<< FLOATING_POINT_POS);
	static constexpr Real INDEX_TO_COORDINATES_FACTOR = 1.0
			/ COORDINATES_TO_INDEX_FACTOR;

	static constexpr coordinates_type dx_ = { 1.0, 1.0, 1.0 };

	static constexpr coordinates_type inv_dx_ = { 1.0, 1.0, 1.0 };

	bool update()
	{

//		for (int i = 0; i < ndims; ++i)
//		{
//			Real L = static_cast<Real>(dimensions_[i]);
//
//			volume_[1UL << (ndims - i - 1)] = 1.0;
//			dual_volume_[7 - (1UL << (ndims - i - 1))] = 1.0;
//			inv_volume_[1UL << (ndims - i - 1)] = 1.0;
//			inv_dual_volume_[7 - (1UL << (ndims - i - 1))] = 1.0;
//
//		}
//

//		volume_[0] = 1;
////		volume_[1] /* 001 */= dx_[0];
////		volume_[2] /* 010 */= dx_[1];
////		volume_[4] /* 100 */= dx_[2];
//
//		volume_[3] /* 011 */= volume_[1] * volume_[2];
//		volume_[5] /* 101 */= volume_[4] * volume_[1];
//		volume_[6] /* 110 */= volume_[2] * volume_[4];
//
//		volume_[7] /* 111 */= volume_[1] * volume_[2] * volume_[4];
//
//		dual_volume_[7] = 1;
////		dual_volume_[6] /* 001 */= dx_[0];
////		dual_volume_[5] /* 010 */= dx_[1];
////		dual_volume_[3] /* 100 */= dx_[2];
//
//		dual_volume_[4] /* 011 */= dual_volume_[6] * dual_volume_[5];
//		dual_volume_[2] /* 101 */= dual_volume_[3] * dual_volume_[6];
//		dual_volume_[1] /* 110 */= dual_volume_[5] * dual_volume_[3];
//
//		dual_volume_[0] /* 111 */= dual_volume_[6] * dual_volume_[5]
//				* dual_volume_[3];
//
//		inv_volume_[0] = 1;
////		inv_volume_[1] /* 001 */= inv_dx_[0];
////		inv_volume_[2] /* 010 */= inv_dx_[1];
////		inv_volume_[4] /* 100 */= inv_dx_[2];
//
//		inv_volume_[3] /* 011 */= inv_volume_[1] * inv_volume_[2];
//		inv_volume_[5] /* 101 */= inv_volume_[4] * inv_volume_[1];
//		inv_volume_[6] /* 110 */= inv_volume_[2] * inv_volume_[4];
//
//		inv_volume_[7] /* 111 */= inv_volume_[1] * inv_volume_[2]
//				* inv_volume_[4];
//
//		inv_dual_volume_[7] = 1;
////		inv_dual_volume_[6] /* 001 */= inv_dx_[0];
////		inv_dual_volume_[5] /* 010 */= inv_dx_[1];
////		inv_dual_volume_[3] /* 100 */= inv_dx_[2];
//
//		inv_dual_volume_[4] /* 011 */= inv_dual_volume_[6]
//				* inv_dual_volume_[5];
//		inv_dual_volume_[2] /* 101 */= inv_dual_volume_[3]
//				* inv_dual_volume_[6];
//		inv_dual_volume_[1] /* 110 */= inv_dual_volume_[5]
//				* inv_dual_volume_[3];
//
//		inv_dual_volume_[0] /* 111 */= inv_dual_volume_[6] * inv_dual_volume_[5]
//				* inv_dual_volume_[3];

		is_valid_ = true;

		return is_valid_;
	}

	std::pair<coordinates_type, coordinates_type> extents() const
	{
		coordinates_type b, e;
		b = 0;
		e = count_;

		return std::move(std::make_pair(b, e));
	}

	static constexpr coordinates_type dx()
	{
		return coordinates_type( { 1.0, 1.0, 1.0 });
	}

	static constexpr Real volume(id_type s)
	{
//		return volume_[node_id(s)];
		return 1.0;
	}

	static constexpr Real inv_volume(id_type s)
	{
		return 1.0;
//		return inv_volume_[node_id(s)];
	}

	static constexpr Real dual_volume(id_type s)
	{
		return 1.0;
//		return dual_volume_[node_id(s)];
	}

	static constexpr Real inv_dual_volume(id_type s)
	{
		return 1.0;
//		return inv_dual_volume_[node_id(s)];
	}

	static constexpr Real cell_volume(id_type s)
	{
		return 1.0;
//		return volume_[1] * volume_[2] * volume_[4];
	}

	static constexpr Real volume(id_type s, std::integral_constant<bool, false>)
	{
		return 1.0;
//		return volume(s);
	}

	static constexpr Real inv_volume(id_type s,
			std::integral_constant<bool, false>)
	{
		return 1.0;
//		return inv_volume(s);
	}

	static constexpr Real inv_dual_volume(id_type s,
			std::integral_constant<bool, false>)
	{
		return 1.0;
//		return inv_dual_volume(s);
	}
	static constexpr Real dual_volume(id_type s,
			std::integral_constant<bool, false>)
	{
		return 1.0;
//		return dual_volume(s);
	}

	static constexpr Real volume(id_type s, std::integral_constant<bool, true>)
	{
		return 1.0;
//		return in_range(s) ? volume(s) : 0.0;
	}

	static constexpr Real inv_volume(id_type s,
			std::integral_constant<bool, true>)
	{
		return 1.0;
//		return in_range(s) ? inv_volume(s) : 0.0;
	}

	static constexpr Real dual_volume(id_type s,
			std::integral_constant<bool, true>)
	{
		return 1.0;
//		return in_range(s) ? dual_volume(s) : 0.0;
	}
	static constexpr Real inv_dual_volume(id_type s,
			std::integral_constant<bool, true>)
	{
		return 1.0;
//		return in_range(s) ? inv_dual_volume(s) : 0.0;
	}

	//! @}

	//! @name Coordinates
	//! @{

	/***
	 *
	 * @param s
	 * @return Coordinates range in [0,1)
	 */
	static constexpr inline coordinates_type index_to_coordinates(
			index_tuple const&idx)
	{

		return std::move(
				coordinates_type(
						{

								static_cast<Real>(static_cast<long>(idx[0])
										- ZERO_INDEX)
										* INDEX_TO_COORDINATES_FACTOR,

								static_cast<Real>(static_cast<long>(idx[1])
										- ZERO_INDEX)
										* INDEX_TO_COORDINATES_FACTOR,

								static_cast<Real>(static_cast<long>(idx[2])
										- ZERO_INDEX)
										* INDEX_TO_COORDINATES_FACTOR

						}));
	}

	static constexpr inline index_tuple coordinates_to_index(
			coordinates_type const &x)
	{
		return std::move(
				index_tuple(
						{

						static_cast<index_type>(static_cast<long>(x[0]
								* COORDINATES_TO_INDEX_FACTOR) + ZERO_INDEX),

						static_cast<index_type>(static_cast<long>(x[1]
								* COORDINATES_TO_INDEX_FACTOR) + ZERO_INDEX),

						static_cast<index_type>(static_cast<long>(x[2]
								* COORDINATES_TO_INDEX_FACTOR) + ZERO_INDEX)

						}));
	}

	static constexpr id_type index_to_id(index_tuple const & x)
	{
		return

		static_cast<id_type>(

		((x[0] & INDEX_MASK)) |

		((x[1] & INDEX_MASK) << (INDEX_DIGITS)) |

		((x[2] & INDEX_MASK) << (INDEX_DIGITS * 2))

		);
	}

	static constexpr index_tuple id_to_index(id_type s)
	{

		return std::move(index_tuple( {

		static_cast<index_type>(s & INDEX_MASK),

		static_cast<index_type>((s >> (INDEX_DIGITS)) & INDEX_MASK),

		static_cast<index_type>((s >> (INDEX_DIGITS * 2)) & INDEX_MASK)

		}));
	}

	static constexpr coordinates_type id_to_coordinates(id_type s)
	{
		return std::move(index_to_coordinates(id_to_index(s)));
	}
	static constexpr id_type coordinates_to_id(coordinates_type const &x)
	{
		return std::move(index_to_id(coordinates_to_index(x)));
	}
	static coordinates_type coordinates_local_to_global(id_type s,
			coordinates_type const &r)
	{
		return (static_cast<coordinates_type>(id_to_coordinates(s) + r));
	}

	static coordinates_type coordinates_local_to_global(
			std::tuple<id_type, coordinates_type> const &z)
	{
		return std::move(
				coordinates_local_to_global(std::get<0>(z), std::get<1>(z)));
	}

	/**
	 *
	 * @param x coordinates \f$ x \in \left[0,1\right)\f$
	 * @param shift
	 * @return s,r  s is the largest grid point not greater than x.
	 *       and  \f$ r \in \left[0,1.0\right) \f$ is the normalize  distance between x and s
	 */

	static std::tuple<id_type, coordinates_type> coordinates_global_to_local(
			coordinates_type const & x, id_type shift = 0UL)
	{
		id_type s = (coordinates_to_id(x) & CELL_ID_MASK) | shift;

		coordinates_type r;

		r = x - id_to_coordinates(s);

		ASSERT(inner_product(r, r) < 1.0);

		return std::move(std::forward_as_tuple(s, r));
	}

	/**
	 *
	 * @param x  coordinates \f$ x \in \left[0,1\right)\f$
	 * @param shift
	 * @return s,r   s is thte conmpact index of nearest grid point
	 *    and  \f$ r \in \left[-0.5,0.5\right) \f$   is the normalize  distance between x and s
	 */
	static std::tuple<id_type, coordinates_type> coordinates_global_to_local_NGP(
			std::tuple<id_type, coordinates_type> const &z)
	{
		auto & x = std::get<1>(z);
		id_type shift = std::get<0>(z);

		index_tuple I = id_to_index(shift >> (FLOATING_POINT_POS - 1));

		coordinates_type r;

		r[0] = x[0] - 0.5 * static_cast<Real>(I[0]);
		r[1] = x[1] - 0.5 * static_cast<Real>(I[1]);
		r[2] = x[2] - 0.5 * static_cast<Real>(I[2]);

		I[0] = static_cast<index_type>(std::floor(r[0] + 0.5));
		I[1] = static_cast<index_type>(std::floor(r[1] + 0.5));
		I[2] = static_cast<index_type>(std::floor(r[2] + 0.5));

		r -= I;

		id_type s = (index_to_id(I)) | shift;

		return std::move(std::make_tuple(s, r));
	}

	//! @}

	//! @name id auxiliary functions
	//! @{
	static constexpr id_type dual(id_type r)
	{
		return (r & (~_DA)) | ((~(r & _DA)) & _DA);

	}
	static constexpr id_type get_cell_id(id_type r)
	{
		return r & CELL_ID_MASK;
	}
	static constexpr id_type node_id(id_type s)
	{

		return (((s >> (INDEX_DIGITS * 2 + FLOATING_POINT_POS - 1)) & 1UL) << 2)
				| (((s >> (INDEX_DIGITS + FLOATING_POINT_POS - 1)) & 1UL) << 1)
				| ((s >> (FLOATING_POINT_POS - 1)) & 1UL);

	}

	static constexpr id_type get_shift(id_type nodeid, id_type h = 0UL)
	{

		return

		(((nodeid & 4UL) >> 2) << (INDEX_DIGITS * 2 + FLOATING_POINT_POS - 1)) |

		(((nodeid & 2UL) >> 1) << (INDEX_DIGITS + FLOATING_POINT_POS - 1)) |

		((nodeid & 1UL) << (FLOATING_POINT_POS - 1));

	}

	static constexpr id_type get_first_node_shift(id_type iform)
	{
//		id_type nid;
//		switch (iform)
//		{
//		case VERTEX:
//			nid = 0;
//			break;
//		case EDGE:
//			nid = 1;
//			break;
//		case FACE:
//			nid = 6;
//			break;
//		case VOLUME:
//			nid = 7;
//			break;
//		}
		// FIXME not complete

		return get_shift(0);
	}

	static constexpr size_t get_num_of_comp_per_cell(size_t iform)
	{

		return (iform == EDGE || iform == FACE) ? 3 : 1;
	}

	static constexpr id_type inverse_roate(id_type r)
	{

		return (r & (~_DA))

		| ((r & (((_DJ | _DK)))) >> INDEX_DIGITS)

		| ((r & (((_DI)))) << (INDEX_DIGITS * 2));

	}

	/**
	 *  rotate vector direction  mask
	 *  (1/2,0,0) => (0,0,1/2) or   (1/2,1/2,0) => (1/2,0,1/2)
	 * @param s
	 * @return
	 */

	static constexpr id_type roate(id_type s)
	{

		return (s & (~(_DA)))

		| ((s & (((_DI | _DJ)))) << INDEX_DIGITS)

		| ((s & (((_DK)))) >> (INDEX_DIGITS * 2));

	}

	static constexpr id_type delta_index(id_type r)
	{
		return (r & _DA);
	}

	static constexpr id_type DI(size_t i, id_type r)
	{
		return (1UL << (INDEX_DIGITS * i + FLOATING_POINT_POS - 1));

	}
	static constexpr id_type delta_index(size_t i, id_type r)
	{
		return DI(i, r) & r;
	}

	/**
	 * Get component number or vector direction
	 * @param s
	 * @return
	 */
	static constexpr id_type component_number(id_type s)
	{
//		id_type res = 0;
//		switch (node_id(s))
//		{
//		case 1:
//		case 6:
//			res = 0;
//			break;
//		case 2:
//		case 5:
//			res = 1;
//			break;
//
//		case 4:
//		case 3:
//			res = 2;
//			break;
//		}
		return (node_id(s) == 1 || node_id(s) == 6) ?
				0 : ((node_id(s) == 2 || node_id(s) == 5) ? 1 : 2);
	}

	static constexpr id_type IForm(id_type r)
	{
//		id_type res = 0;
//		switch (node_id(r))
//		{
//		case 0:
//			res = VERTEX;
//			break;
//		case 1:
//		case 2:
//		case 4:
//			res = EDGE;
//			break;
//
//		case 3:
//		case 5:
//		case 6:
//			res = FACE;
//			break;
//
//		case 7:
//			res = VOLUME;
//		}
		// FIXME : NOT complete;

		return (node_id(r) == 0) ? VERTEX : ((node_id(r) == 7) ? VOLUME : FACE)

		;
	}
	//! @}

	template<size_t IForm> struct Range;

//	template<size_t IForm>
//	static Range<IForm> make_range(nTuple<size_t, ndims> const & b,
//			nTuple<size_t, ndims> const &e)
//	{
//		index_tuple b1, e1;
//		b1 = (b << FLOATING_POINT_POS) + ZERO_INDEX;
//		e1 = (e << FLOATING_POINT_POS) + ZERO_INDEX;
//		CHECK(b1);
//		CHECK(e1);
//		return std::move(Range<IForm>(b1, e1));
//	}
//	template<size_t IForm>
//	static Range<IForm> make_range(coordinates_type const & b,
//			coordinates_type const &e)
//	{
//		return std::move(
//				Range<IForm>(coordinates_to_index(b), coordinates_to_index(e)));
//	}

	template<size_t IForm>
	constexpr Range<IForm> range() const
	{
		return Range<IForm>(local_inner_begin_, local_inner_end_);
	}

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
	constexpr Range<IFORM> select() const
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
	constexpr Range<IFORM> select(coordinates_type const & xmin,
			coordinates_type const &xmax) const
	{
		return std::move(Range<IFORM>());
	}

	template<size_t IFORM>
	constexpr Range<IFORM> select_outer() const
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
//		nTuple<index_type, ndims> d = (id_to_index(s) >> FLOATING_POINT_POS)
//				- local_outer_begin_;
//
//		size_t res =
//
//		mod_(d[0], (local_outer_count_[0])) * hash_strides_[0] +
//
//		mod_(d[1], (local_outer_count_[1])) * hash_strides_[1] +
//
//		mod_(d[2], (local_outer_count_[2])) * hash_strides_[2];
//
//		switch (node_id(s))
//		{
//		case 4:
//		case 3:
//			res = ((res << 1) + res);
//			break;
//		case 2:
//		case 5:
//			res = ((res << 1) + res) + 1;
//			break;
//		case 1:
//		case 6:
//			res = ((res << 1) + res) + 2;
//			break;
//		}
//
//		return res;

		return 0;

	}
	/**@}*/

	/**
	 * @name Neighgour
	 * @{
	 */

	static size_t get_vertices(id_type s, id_type *v)
	{
		size_t res = 0;
		switch (IForm(s))
		{
		case VERTEX:
			res = get_vertices(std::integral_constant<size_t, VERTEX>(), s, v);
			break;
		case EDGE:
			res = get_vertices(std::integral_constant<size_t, EDGE>(), s, v);
			break;
		case FACE:
			res = get_vertices(std::integral_constant<size_t, FACE>(), s, v);
			break;
		case VOLUME:
			res = get_vertices(std::integral_constant<size_t, VOLUME>(), s, v);
			break;
		}
		return res;
	}

	template<size_t IFORM>
	static size_t get_vertices(std::integral_constant<size_t, IFORM>, id_type s,
			id_type *v)
	{
		return get_adjacent_cells(std::integral_constant<size_t, IFORM>(),
				std::integral_constant<size_t, VERTEX>(), s, v);
	}

	template<size_t I>
	static inline size_t get_adjacent_cells(std::integral_constant<size_t, I>,
			std::integral_constant<size_t, I>, id_type s, id_type *v)
	{
		v[0] = s;
		return 1;
	}

	static inline size_t get_adjacent_cells(
			std::integral_constant<size_t, EDGE>,
			std::integral_constant<size_t, VERTEX>, id_type s, id_type *v)
	{
		v[0] = s + delta_index(s);
		v[1] = s - delta_index(s);
		return 2;
	}

	static inline size_t get_adjacent_cells(
			std::integral_constant<size_t, FACE>,
			std::integral_constant<size_t, VERTEX>, id_type s, id_type *v)
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

	static inline size_t get_adjacent_cells(
			std::integral_constant<size_t, VOLUME>,
			std::integral_constant<size_t, VERTEX>, id_type s, id_type *v)
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

	static inline size_t get_adjacent_cells(
			std::integral_constant<size_t, VERTEX>,
			std::integral_constant<size_t, EDGE>, id_type s, id_type *v)
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

	static inline size_t get_adjacent_cells(
			std::integral_constant<size_t, FACE>,
			std::integral_constant<size_t, EDGE>, id_type s, id_type *v)
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

	static inline size_t get_adjacent_cells(
			std::integral_constant<size_t, VOLUME>,
			std::integral_constant<size_t, EDGE>, id_type s, id_type *v)
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

	static inline size_t get_adjacent_cells(
			std::integral_constant<size_t, VERTEX>,
			std::integral_constant<size_t, FACE>, id_type s, id_type *v)
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

	static inline size_t get_adjacent_cells(
			std::integral_constant<size_t, EDGE>,
			std::integral_constant<size_t, FACE>, id_type s, id_type *v)
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

	static inline size_t get_adjacent_cells(
			std::integral_constant<size_t, VOLUME>,
			std::integral_constant<size_t, FACE>, id_type s, id_type *v)
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

	static inline size_t get_adjacent_cells(
			std::integral_constant<size_t, VERTEX>,
			std::integral_constant<size_t, VOLUME>, id_type s, id_type *v)
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

	static inline size_t get_adjacent_cells(
			std::integral_constant<size_t, EDGE>,
			std::integral_constant<size_t, VOLUME>, id_type s, id_type *v)
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

	static inline size_t get_adjacent_cells(
			std::integral_constant<size_t, FACE>,
			std::integral_constant<size_t, VOLUME>, id_type s, id_type *v)
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
	/**@}*/

	template<typename TV>
	static TV sample_(std::integral_constant<size_t, VERTEX>, size_t s,
			TV const &v)
	{
		return v;
	}

	template<typename TV>
	static TV sample_(std::integral_constant<size_t, VOLUME>, size_t s,
			TV const &v)
	{
		return v;
	}

	template<typename TV>
	static TV sample_(std::integral_constant<size_t, EDGE>, size_t s,
			nTuple<TV, 3> const &v)
	{
		return v[component_number(s)];
	}

	template<typename TV>
	static TV sample_(std::integral_constant<size_t, FACE>, size_t s,
			nTuple<TV, 3> const &v)
	{
		return v[component_number(s)];
	}

	template<size_t IFORM, typename TV>
	static TV sample_(std::integral_constant<size_t, IFORM>, size_t s,
			TV const & v)
	{
		return v;
	}

	template<size_t IFORM, typename ...Args>
	static auto sample(Args && ... args)
	DECL_RET_TYPE((sample_(std::integral_constant<size_t, IFORM>(),
							std::forward<Args>(args)...)))

};

class op_split
{

};

template<size_t IFORM>
struct StructuredMesh::Range
{

	struct const_iterator;

	index_tuple begin_, end_;

	Range()
	{
	}

	Range(index_tuple const & b, index_tuple const& e) :
			begin_(b), end_(e)
	{
	}

	Range(Range const & that) :
			begin_(that.begin_), end_(that.end_)
	{
	}
	Range(Range & that, op_split) :
			begin_(that.begin_), end_(that.end_)
	{
	}
	~Range()
	{
	}

	const_iterator begin() const
	{
		return const_iterator(begin_, end_);
	}

	const_iterator end() const
	{
		index_tuple s = end_;
		s -= 1;
		const_iterator res(begin_, end_, s);
		++res;
		return std::move(res);
	}

};

template<size_t IFORM>
struct StructuredMesh::Range<IFORM>::const_iterator
{
	typedef id_type value_type;

	index_tuple begin_, end_;

	index_tuple self_;

	id_type shift_ = (IFORM == VERTEX) ? (0UL) :

	((IFORM == EDGE) ? (_DI) :

	((IFORM == FACE) ? (_DJ | _DK) :

	(_DI | _DJ | _DK)));

	const_iterator(const_iterator const & r) :
			shift_(r.shift_), self_(r.self_), begin_(r.begin_), end_(r.end_)
	{
	}
	const_iterator(const_iterator && r) :
			shift_(r.shift_), self_(r.self_), begin_(r.begin_), end_(r.end_)
	{
	}
	const_iterator(index_tuple const & b, index_tuple const &e) :
			self_(b), begin_(b), end_(e)
	{
	}
	const_iterator(index_tuple const & b, index_tuple const &e,
			index_tuple const & s) :
			self_(s), begin_(b), end_(e)
	{
	}

	~const_iterator()
	{
	}

	bool operator==(const_iterator const & rhs) const
	{
		return self_ == rhs.self_ && shift_ == rhs.shift_;
	}

	bool operator!=(const_iterator const & rhs) const
	{
		return !(this->operator==(rhs));
	}

	value_type operator*() const
	{
		return index_to_id(self_) | shift_;
	}

	const_iterator & operator ++()
	{
		next();
		return *this;
	}
	const_iterator operator ++(int) const
	{
		const_iterator res(*this);
		++res;
		return std::move(res);
	}

//	const_iterator & operator --()
//	{
//		prev();
//		return *this;
//	}
//
//	const_iterator operator --(int) const
//	{
//		const_iterator res(*this);
//		--res;
//		return std::move(res);
//	}
private:
#ifndef USE_FORTRAN_ORDER_ARRAY
	static constexpr size_t ARRAY_ORDER = C_ORDER;
#else
	static constexpr size_t ARRAY_ORDER=FOTRAN_ORDER;
#endif

	void next()
	{

		if (roate_shift(std::integral_constant<size_t, IFORM>()))
		{

#ifndef USE_FORTRAN_ORDER_ARRAY
			self_[ndims - 1] += D_INDEX;

			for (int i = ndims - 1; i > 0; --i)
			{
				if (self_[i] >= end_[i])
				{
					self_[i] = begin_[i];
					self_[i - 1] += D_INDEX;
				}
			}
#else
			self_[0]+=D_INDEX;

			for (int i = 0; i < ndims - 1; ++i)
			{
				if (self_[i] >= end_[i])
				{
					self_[i] = begin_[i];
					self_[i + 1]+=D_INDEX;
				}
			}
#endif
		}
	}

	void prev()
	{
		if (inv_roate_shift(std::integral_constant<size_t, IFORM>()))
		{
#ifndef USE_FORTRAN_ORDER_ARRAY

			if (self_[ndims - 1] > begin_[ndims - 1])
				self_[ndims - 1] -= D_INDEX;

			for (int i = ndims - 1; i > 0; --i)
			{
				if (self_[i] <= begin_[i])
				{
					self_[i] = end_[i] - 1;

					if (self_[i - 1] > begin_[i - 1])
						self_[i - 1] -= D_INDEX;
				}
			}

#else

			self_[0]-=D_INDEX;
			for (int i = 0; i < ndims; ++i)
			{
				if (self_[i] < begin_[i])
				{
					self_[i] = end_[i] - 1;
					self_[i + 1]-=D_INDEX;
				}
			}

#endif //USE_FORTRAN_ORDER_ARRAY
		}
	}
	constexpr bool roate_shift(std::integral_constant<size_t, VERTEX>) const
	{
		return true;
	}
	constexpr bool roate_shift(std::integral_constant<size_t, VOLUME>) const
	{
		return true;
	}

	constexpr bool inv_roate_shift(std::integral_constant<size_t, VERTEX>) const
	{
		return true;
	}
	constexpr bool inv_roate_shift(std::integral_constant<size_t, VOLUME>) const
	{
		return true;
	}
	bool roate_shift(std::integral_constant<size_t, EDGE>)
	{
		shift_ = roate(shift_);
		return node_id(shift_) == 1;
	}

	bool roate_shift(std::integral_constant<size_t, FACE>)
	{
		shift_ = roate(shift_);
		return node_id(shift_) == 6;
	}
	bool inv_roate_shift(std::integral_constant<size_t, EDGE>)
	{
		shift_ = inverse_roate(shift_);
		return node_id(shift_) == 4;
	}

	bool inv_roate_shift(std::integral_constant<size_t, FACE>)
	{
		shift_ = inverse_roate(shift_);
		return node_id(shift_) == 3;
	}
};
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
