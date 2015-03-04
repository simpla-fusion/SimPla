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
#include "../../../numeric/geometric_algorithm.h"
#include "../../mesh_common.h"

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
template<size_t NDIMS = 3>
struct StructuredMesh_
{

	typedef StructuredMesh_ this_type;

	static constexpr size_t ndims = NDIMS;

	typedef unsigned long index_type;

	typedef nTuple<index_type, ndims> index_tuple;

	typedef nTuple<Real, ndims> coordinates_type;

	typedef unsigned long id_type;

	static constexpr size_t MAX_NUM_NEIGHBOUR_ELEMENT = 12;
	static constexpr size_t MAX_NUM_VERTEX_PER_CEL = 8;
	static constexpr size_t DEFAULT_GHOSTS_WIDTH = 3;

	template<size_t IForm> struct const_iterator;

	template<size_t IForm> struct Range;

private:
	bool is_valid_ = false;
public:

	//***************************************************************************************************

	StructuredMesh_()
	{
	}
	StructuredMesh_(nTuple<size_t, ndims> const &dims)
	{
		dimensions(&dims[0]);
	}
	virtual ~StructuredMesh_()
	{
	}

	this_type & operator=(const this_type&) = delete;

	StructuredMesh_(const this_type&) = delete;

	void swap(StructuredMesh_ & rhs) = delete;

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
			dimensions(&d[0]);

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

	/**
	 * @name  Data Shape
	 * @{
	 **/

	index_tuple m_dimensions_;

	index_tuple m_local_outer_begin_, m_local_outer_end_, m_local_outer_count_;

	index_tuple m_local_inner_begin_, m_local_inner_end_, m_local_inner_count_;

	index_tuple m_ghost_width;

	index_tuple m_hash_strides_;

//	coordinates_type m_xmin_, m_xmax_;
//
//	static constexpr coordinates_type m_dx_ = { 1, 1, 1 };

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

	static constexpr index_type AXIS_ORIGIN = 1 << (INDEX_DIGITS - 1);

	void dimensions(index_type const * d, index_type const * gw = nullptr)
	{
		m_dimensions_ = d;

		m_local_outer_count_ = (m_dimensions_ << FLOATING_POINT_POS);

		m_local_outer_begin_ = AXIS_ORIGIN;

		m_local_outer_end_ = (m_dimensions_ << FLOATING_POINT_POS)
				+ AXIS_ORIGIN;

		m_local_inner_count_ = (m_dimensions_ << FLOATING_POINT_POS);

		m_local_inner_begin_ = AXIS_ORIGIN;

		m_local_inner_end_ = (m_dimensions_ << FLOATING_POINT_POS)
				+ AXIS_ORIGIN;
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
		m_hash_strides_[2] = 1;
		m_hash_strides_[1] = d[2] * m_hash_strides_[2];
		m_hash_strides_[0] = d[1] * m_hash_strides_[1];

//		m_xmin_ = 0;
//		m_xmax_ = m_dimensions_;

		update();

	}

	index_tuple const & dimensions() const
	{
		return m_dimensions_;
	}

	bool check_memory_bounds(id_type s) const
	{
//		auto idx = id_to_index(s) >> FLOATING_POINT_POS;

		return true;

//		idx[0] >= m_local_outer_begin_[0]
//
//		&& idx[0] < m_local_outer_end_[0]
//
//		&& idx[1] >= m_local_outer_begin_[1]
//
//		&& idx[1] < m_local_outer_end_[1]
//
//		&& idx[2] >= m_local_outer_begin_[2]
//
//		&& idx[2] < m_local_outer_end_[2]

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

		g_dims = m_dimensions_;
		g_offset = m_local_inner_begin_ >> FLOATING_POINT_POS;
		g_count = m_local_inner_count_ >> FLOATING_POINT_POS;
		g_gw = m_ghost_width;

		if (IForm == EDGE || IForm == FACE)
		{
			g_dims[rank] = 3;
			g_offset[rank] = 0;
			g_count[rank] = 3;
			g_gw[rank] = 0;
			++rank;
		}
		g_dims += g_gw * 2;
		g_offset += g_gw;
		DataSpace res(rank, &g_dims[0]);
		res.select_hyperslab(&g_offset[0], nullptr, &g_count[0], nullptr);

		return std::move(res);
	}

	/**
	 *   @name Geometry
	 *   For For uniform structured grid, the volume of cell is 1.0
	 *   and dx=1.0
	 *   @{
	 */

	bool update()
	{
		is_valid_ = true;

		return is_valid_;
	}

//! @name Coordinates
//! @{

	static constexpr Real COORDINATES_TO_INDEX_FACTOR =
			static_cast<Real>(FLOATING_POINT_FACTOR);

	static constexpr Real INDEX_TO_COORDINATES_FACTOR = 1.0
			/ COORDINATES_TO_INDEX_FACTOR;

	static constexpr coordinates_type index_to_coordinates(
			index_tuple const&idx)
	{

		return std::move(
				coordinates_type(
						{

						static_cast<Real>(static_cast<long>(idx[0])
								- AXIS_ORIGIN) * INDEX_TO_COORDINATES_FACTOR,

						static_cast<Real>(static_cast<long>(idx[1])
								- AXIS_ORIGIN) * INDEX_TO_COORDINATES_FACTOR,

						static_cast<Real>(static_cast<long>(idx[2])
								- AXIS_ORIGIN) * INDEX_TO_COORDINATES_FACTOR

						}));
	}

	static constexpr index_tuple coordinates_to_index(coordinates_type const &x)
	{
		return std::move(
				index_tuple(
						{

						static_cast<index_type>(static_cast<long>(x[0]
								* COORDINATES_TO_INDEX_FACTOR) + AXIS_ORIGIN),

						static_cast<index_type>(static_cast<long>(x[1]
								* COORDINATES_TO_INDEX_FACTOR) + AXIS_ORIGIN),

						static_cast<index_type>(static_cast<long>(x[2]
								* COORDINATES_TO_INDEX_FACTOR) + AXIS_ORIGIN)

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

	static constexpr coordinates_type coordinates_local_to_global(id_type s,
			coordinates_type const &r)
	{
		return static_cast<coordinates_type>(id_to_coordinates(s) + r);
	}

	static constexpr coordinates_type coordinates_local_to_global(
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
	static constexpr id_type m_first_node_shift_[] = { 0, 1, 6, 7 };

	static constexpr id_type get_first_node_shift(id_type iform)
	{

		return get_shift(m_first_node_shift_[iform]);
	}
	static constexpr size_t m_num_of_comp_per_cell_[4] = { 1, 3, 3, 1 };

	static constexpr size_t get_num_of_comp_per_cell(size_t iform)
	{
		return m_num_of_comp_per_cell_[iform];
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

	static constexpr id_type m_component_number_[] = { 0,  // 000
			0, // 001
			1, // 010
			2, // 011
			2, // 100
			1, // 101
			0, // 110
			0 };

	static constexpr id_type component_number(id_type s)
	{
		return m_component_number_[node_id(s)];
	}

	static constexpr id_type m_iform_[] = { //

			VERTEX, // 000
					EDGE, // 001
					EDGE, // 010
					FACE, // 011
					EDGE, // 100
					FACE, // 101
					FACE, // 110
					VOLUME // 111
			};

	static constexpr id_type IForm(id_type r)
	{
		return m_iform_[node_id(r)];
	}
//! @}

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
		return (Range<IFORM>(m_local_inner_begin_, m_local_inner_end_));
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
					m_local_inner_begin_, m_local_inner_end_))

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
		return std::move(Range<IFORM>(m_local_outer_begin_, m_local_outer_end_));
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
	DECL_RET_TYPE (select_rectangle_<IFORM>( b, e, m_local_outer_begin_,
					m_local_outer_end_))

	template<size_t IFORM>
	auto select_inner(index_tuple const & b, index_tuple const & e) const
	DECL_RET_TYPE (select_rectangle_<IFORM>( b, e, m_local_inner_begin_,
					m_local_inner_end_))

	/**  @} */
	/**
	 *  @name Hash
	 *  @{
	 */

	template<size_t IFORM>
	size_t max_hash() const
	{
		return NProduct(m_local_outer_count_)
				* ((IFORM == EDGE || IFORM == FACE) ? 3 : 1);
	}

	static index_type mod_(index_type a, index_type L)
	{
		return (a + L) % L;
	}

	size_t hash(id_type s) const
	{

		nTuple<index_type, ndims> d = (id_to_index(s) - m_local_outer_begin_)
				>> FLOATING_POINT_POS;

		size_t res =

		mod_(d[0], (m_local_outer_count_[0])) * m_hash_strides_[0] +

		mod_(d[1], (m_local_outer_count_[1])) * m_hash_strides_[1] +

		mod_(d[2], (m_local_outer_count_[2])) * m_hash_strides_[2];

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

};

using StructuredMesh=StructuredMesh_<3>;
/**
 * Solve problem: Undefined reference to static constexpr char[]
 * http://stackoverflow.com/questions/22172789/passing-a-static-constexpr-variable-by-universal-reference
 */
template<size_t NDIMS> constexpr typename StructuredMesh_<NDIMS>::id_type
StructuredMesh_<NDIMS>::m_component_number_[];
template<size_t NDIMS> constexpr typename StructuredMesh_<NDIMS>::id_type
StructuredMesh_<NDIMS>::m_iform_[];
template<size_t NDIMS> constexpr typename StructuredMesh_<NDIMS>::id_type
StructuredMesh_<NDIMS>::m_first_node_shift_[];

template<size_t NDIMS> constexpr size_t StructuredMesh_<NDIMS>::ndims;
template<size_t NDIMS> constexpr size_t StructuredMesh_<NDIMS>::MAX_NUM_NEIGHBOUR_ELEMENT;
template<size_t NDIMS> constexpr size_t StructuredMesh_<NDIMS>::MAX_NUM_VERTEX_PER_CEL;
template<size_t NDIMS> constexpr size_t StructuredMesh_<NDIMS>::DEFAULT_GHOSTS_WIDTH;

template<size_t NDIMS> constexpr typename StructuredMesh_<NDIMS>::index_type
StructuredMesh_<NDIMS>::FLOATING_POINT_FACTOR;
template<size_t NDIMS> constexpr typename StructuredMesh_<NDIMS>::index_type
StructuredMesh_<NDIMS>::AXIS_ORIGIN;
template<size_t NDIMS> constexpr size_t StructuredMesh_<NDIMS>::FULL_DIGITS;
template<size_t NDIMS> constexpr size_t StructuredMesh_<NDIMS>::INDEX_DIGITS;
template<size_t NDIMS> constexpr size_t StructuredMesh_<NDIMS>::FLOATING_POINT_POS;
template<size_t NDIMS> constexpr size_t StructuredMesh_<NDIMS>::INDEX_MASK;
template<size_t NDIMS> constexpr size_t StructuredMesh_<NDIMS>::D_INDEX;
template<size_t NDIMS> constexpr size_t StructuredMesh_<NDIMS>::_DK;
template<size_t NDIMS> constexpr size_t StructuredMesh_<NDIMS>::_DJ;
template<size_t NDIMS> constexpr size_t StructuredMesh_<NDIMS>::_DI;
template<size_t NDIMS> constexpr size_t StructuredMesh_<NDIMS>::_DA;
template<size_t NDIMS> constexpr size_t StructuredMesh_<NDIMS>::CELL_ID_MASK_;
template<size_t NDIMS> constexpr size_t StructuredMesh_<NDIMS>::CELL_ID_MASK;

template<size_t NDIMS> constexpr Real StructuredMesh_<NDIMS>::COORDINATES_TO_INDEX_FACTOR;
template<size_t NDIMS> constexpr Real StructuredMesh_<NDIMS>::INDEX_TO_COORDINATES_FACTOR;

template<size_t NDIMS>
template<size_t IFORM>
struct StructuredMesh_<NDIMS>::Range
{

	struct const_iterator;

	index_tuple begin_, end_;

	Range()
	{
	}

	Range(index_tuple const & b, index_tuple const& e)
			: begin_(b), end_(e)
	{
	}

	Range(Range const & that)
			: begin_(that.begin_), end_(that.end_)
	{
	}
	~Range()
	{
	}

	const_iterator begin() const
	{
		return const_iterator(begin_, end_, begin_);
	}

	const_iterator end() const
	{
		auto t = end_ - D_INDEX;
		const_iterator res(begin_, end_, t);
		++res;

		return std::move(res);
	}

};
template<size_t NDIMS>

template<size_t IFORM>
struct StructuredMesh_<NDIMS>::Range<IFORM>::const_iterator
{
	typedef id_type value_type;

	index_tuple begin_, end_;

	index_tuple self_;

	id_type shift_ = (IFORM == VERTEX) ? (0UL) :

	((IFORM == EDGE) ? (_DI) :

	((IFORM == FACE) ? (_DJ | _DK) :

	(_DI | _DJ | _DK)));

	const_iterator(const_iterator const & r)
			: shift_(r.shift_), self_(r.self_), begin_(r.begin_), end_(r.end_)
	{
	}
	const_iterator(const_iterator && r)
			: shift_(r.shift_), self_(r.self_), begin_(r.begin_), end_(r.end_)
	{
	}

	const_iterator(index_tuple const & b, index_tuple const &e,
			index_tuple const &s)
			: self_(s), begin_(b), end_(e)
	{
	}
	~const_iterator()
	{
	}

	bool operator==(const_iterator const & rhs) const
	{
		return (self_[0] == rhs.self_[0]) && (self_[1] == rhs.self_[1])
				&& (self_[2] == rhs.self_[2]) && (shift_ == rhs.shift_);
	}

	constexpr bool operator!=(const_iterator const & rhs) const
	{
		return !(this->operator==(rhs));
	}

	constexpr value_type operator*() const
	{
		return (index_to_id(self_)) | shift_;
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
//
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
	static constexpr size_t ARRAY_ORDER = C_ORDER;

	void next()
	{

		if (roate_shift(std::integral_constant<size_t, IFORM>()))
		{
			self_[ndims - 1] += D_INDEX;

			for (int i = ndims - 1; i > 0; --i)
			{
				if (self_[i] >= end_[i])
				{
					self_[i] = begin_[i];
					self_[i - 1] += D_INDEX;
				}
			}
		}
	}

	void prev()
	{
		if (inv_roate_shift(std::integral_constant<size_t, IFORM>()))
		{

			if (self_[ndims - 1] > begin_[ndims - 1])
				--self_[ndims - 1];

			for (int i = ndims - 1; i > 0; --i)
			{
				if (self_[i] <= begin_[i])
				{
					self_[i] = end_[i] - 1;

					if (self_[i - 1] > begin_[i - 1])
						--self_[i - 1];
				}
			}

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

#endif /* MESH_STRUCTURED_H_ */
