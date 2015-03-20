/**
 * @file mesh_ids.h
 *
 * @date 2015年3月19日
 * @author salmon
 */

#ifndef CORE_MESH_MESH_IDS_H_
#define CORE_MESH_MESH_IDS_H_

#include <stddef.h>
#include <algorithm>
#include <limits>

#include "../gtl/ntuple.h"
#include "../gtl/primitives.h"
#include "../gtl/type_traits.h"

namespace simpla
{
enum ManifoldTypeID
{
	VERTEX = 0,

	EDGE = 1,

	FACE = 2,

	VOLUME = 3
};

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
template<size_t NDIMS = 3, size_t AXIS_FLAG = 0>
struct MeshIDs_
{

	static constexpr size_t ndims = NDIMS;

	static constexpr size_t FULL_DIGITS = std::numeric_limits<size_t>::digits;

	static constexpr size_t INDEX_DIGITS = (FULL_DIGITS
			- CountBits<FULL_DIGITS>::n) / 3;

	static constexpr size_t FLOATING_POINT_POS = 4;

	static constexpr size_t FLOATING_POINT_FACTOR = 1 << FLOATING_POINT_POS;

	static constexpr size_t INDEX_ZERO = 1UL
			<< (INDEX_DIGITS - FLOATING_POINT_POS - 1);

	static constexpr size_t ID_ZERO = (INDEX_ZERO << (FLOATING_POINT_POS))
			| (INDEX_ZERO << (FLOATING_POINT_POS + INDEX_DIGITS))
			| (INDEX_ZERO << (FLOATING_POINT_POS + INDEX_DIGITS * 2));

	static constexpr Real COORDINATE_ZERO = static_cast<Real>(INDEX_ZERO);

	static constexpr Real COORDINATES_TO_INDEX_FACTOR = static_cast<Real>(1
			<< FLOATING_POINT_POS);

	static constexpr Real INDEX_TO_COORDINATES_FACTOR = 1.0
			/ COORDINATES_TO_INDEX_FACTOR;

	static constexpr size_t INDEX_MASK = (1UL << (INDEX_DIGITS + 1)) - 1;

	static constexpr size_t D_INDEX = (1UL << (FLOATING_POINT_POS));

	static constexpr size_t _DK = D_INDEX << (INDEX_DIGITS * 2);

	static constexpr size_t _DJ = D_INDEX << (INDEX_DIGITS);

	static constexpr size_t _DI = D_INDEX;

	static constexpr size_t _DA = _DI | _DJ | _DK;

	static constexpr size_t CELL_ID_MASK_ = //
			(((1UL << (INDEX_DIGITS - FLOATING_POINT_POS)) - 1)
					<< (FLOATING_POINT_POS)) & INDEX_MASK;

	static constexpr size_t CELL_ID_MASK =

	(CELL_ID_MASK_ << (INDEX_DIGITS * 2))

	| (CELL_ID_MASK_ << (INDEX_DIGITS))

	| (CELL_ID_MASK_);

	static constexpr size_t SUB_CELL_ID_MASK_ = 1 << (FLOATING_POINT_POS - 1);

	static constexpr size_t SUB_CELL_ID_MASK =

	(SUB_CELL_ID_MASK_ << (INDEX_DIGITS * 2))

	| (SUB_CELL_ID_MASK_ << (INDEX_DIGITS))

	| (SUB_CELL_ID_MASK_);

	static constexpr size_t ID_MASK =

	(((AXIS_FLAG & 1UL) == 0) ? (INDEX_MASK) : 0UL)

	| (((AXIS_FLAG & 2UL) == 0) ? (INDEX_MASK << INDEX_DIGITS) : 0UL)

	| (((AXIS_FLAG & 4UL) == 0) ? (INDEX_MASK << (INDEX_DIGITS * 2)) : 0UL);

	typedef size_t id_type;
	typedef nTuple<Real, 3> coordinates_type;
	typedef nTuple<long, 3> index_tuple;

	static constexpr size_t m_sub_node_id_[4][3] = { 0, 0, 0, //
			_DI >> 1, _DJ >> 1, _DK >> 1, //
			(_DJ | _DK) >> 1, //
			(_DK | _DI) >> 1, //
			(_DI | _DJ) >> 1, //
			(_DA) >> 1, //
			(_DA) >> 1, //
			(_DA) >> 1 };

	static constexpr coordinates_type m_sub_node_num_[4][3] = { //

			0, 0, 0, /*VERTEX*/

			1, 2, 4, /*EDGE*/

			6, 5, 3, /*FACE*/

			7, 7, 7 /*VOLUME*/

			};

	static constexpr coordinates_type m_sub_node_shift_[4][3] = {

	{ 0, 0, 0, 0, 0, 0, 0, 0, 0 }, /*VERTEX*/

	{ 0.5, 0, 0, 0, 0.5, 0, 0, 0, 0.5 }, /*EDGE*/

	{ 0, 0.5, 0.5, 0.5, 0, 0.5, 0.5, 0.5, 0 }, /*FACE*/

	{ 0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5 } /*VOLUME*/

	};

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
	static constexpr coordinates_type id_to_coordinates(id_type s)
	{
		return std::move(coordinates_type { //

				static_cast<Real>(s & INDEX_MASK) //
				* INDEX_TO_COORDINATES_FACTOR - COORDINATE_ZERO,

				static_cast<Real>((s >> INDEX_DIGITS) & INDEX_MASK) //
				* INDEX_TO_COORDINATES_FACTOR - COORDINATE_ZERO,

				static_cast<Real>((s >> (INDEX_DIGITS * 2)) & INDEX_MASK) //
				* INDEX_TO_COORDINATES_FACTOR - COORDINATE_ZERO

				}

				);
	}

	template<size_t IFORM = 0>
	static constexpr id_type id(size_t i, size_t j, size_t k, size_t n = 0)
	{
		return ((i + INDEX_ZERO) << FLOATING_POINT_POS)
				| ((j + INDEX_ZERO) << (FLOATING_POINT_POS + INDEX_DIGITS))
				| ((k + INDEX_ZERO) << (FLOATING_POINT_POS + INDEX_DIGITS * 2))
				| m_sub_node_id_[IFORM][n];
	}

	static constexpr id_type coordinates_to_id(coordinates_type const &x)
	{
		return

		(static_cast<size_t>((x[0] + COORDINATE_ZERO)
				* COORDINATES_TO_INDEX_FACTOR) & INDEX_MASK)

		| ((static_cast<size_t>((x[1] + COORDINATE_ZERO)  //
		* COORDINATES_TO_INDEX_FACTOR) & INDEX_MASK) << (INDEX_DIGITS))

		| ((static_cast<size_t>((x[2] + COORDINATE_ZERO)  //
		* COORDINATES_TO_INDEX_FACTOR) & INDEX_MASK) << (INDEX_DIGITS * 2));
	}

	template<size_t FLOAT_POS = FLOATING_POINT_POS>
	index_tuple id_to_index(id_type const & s)
	{
		return std::move(
				index_tuple(
						{

						static_cast<long>((s & INDEX_MASK) >> FLOAT_POS)
								- static_cast<long>(INDEX_ZERO >> FLOAT_POS),

						static_cast<long>(((s >> (INDEX_DIGITS)) & INDEX_MASK)
								>> FLOAT_POS)
								- static_cast<long>(INDEX_ZERO >> FLOAT_POS),

						static_cast<long>(((s >> (INDEX_DIGITS * 2))
								& INDEX_MASK) >> FLOAT_POS)
								- static_cast<long>(INDEX_ZERO >> FLOAT_POS)

						}));
	}

	id_type dual(id_type const & s) const
	{
		//TODO
	}

	template<size_t I_FLOAT_POS = FLOATING_POINT_POS>
	struct id_hasher
	{

	private:
		index_tuple m_offset_;
		nTuple<size_t, ndims + 1> m_dimensions_;
		nTuple<size_t, ndims + 1> m_strides_
	public:
		typedef id_hasher<I_FLOAT_POS> this_type;

		template<typename T0>
		id_hasher(T0 const & d)
		{
			m_dimensions_ = d;
			m_offset_ = 0;
			deploy();
		}

		template<typename T0, typename T1>
		id_hasher(T0 const & d, T1 const & offset)
		{
			m_dimensions_ = d;
			m_offset_ = offset;
			deploy();
		}

		id_hasher(this_type const & other)
				: m_dimensions_(other.m_dimensions_), m_offset_(other.m_offset_)
		{
			deploy();
		}
		id_hasher(this_type && other)
				: m_dimensions_(other.m_dimensions_), m_offset_(other.m_offset_)
		{
			deploy();
		}
		~id_hasher()
		{
		}

		void swap(this_type & other)
		{
			std::swap(m_offset_, other.m_offset_);
			std::swap(m_dimensions_, other.m_dimensions_);
			std::swap(m_strides_, other.m_strides_);
		}
		this_type & operator=(this_type const & other)
		{
			this_type(other).swap(*this);
			return *this;
		}

		template<size_t IFORM = VERTEX>
		size_t max_hash() const
		{
			return (m_dimensions_[0] * m_strides_[0])
					* ((IFORM == EDGE || IFORM == FACE) ? 3 : 1);
		}

		void deploy()
		{
			m_strides_[ndims - 1] = 1;

			if (ndims > 1)
			{
				for (int i = ndims - 2; i >= 0; --i)
				{
					m_strides_[i] = m_dimensions_[i + 1] * m_strides_[i + 1];
				}
			}
		}
		constexpr size_t operator()(id_type const & s) const
		{
			return inner_product(
					(id_to_index<I_FLOAT_POS>(s) + m_dimensions_ - m_offset_)
							% m_dimensions_, m_strides_);
		}

		template<size_t IFORM>
		constexpr size_t hash(id_type const & s) const
		{
			return inner_product(
					(id_to_index<I_FLOAT_POS>(s) + m_dimensions_ - m_offset_)
							% m_dimensions_, m_strides_);
		}

	};

	typedef SpHashContainer<id_type, Real,
			id_hasher<MeshIDs::FLOATING_POINT_POS - 1>> volume_container;

	template<typename ...Args>
	volume_container make_volume_container(Args&& ...args) const
	{
		id_hasher<MeshIDs::FLOATING_POINT_POS - 1> hasher(
				std::forward<Args>(args)...);

		return std::move(volume_container(hasher.max_hash(), std::move(hasher)));
	}

	//**************************************************************************
	/**
	 * @name Neighgour
	 * @{
	 */
//	template<size_t IFORM>
//	static size_t get_vertices(id_type s, id_type *v)
//	{
//		size_t res = 0;
//		switch (IForm(s))
//		{
//		case VERTEX:
//			res = get_vertices(std::integral_constant<size_t, VERTEX>(), s, v);
//			break;
//		case EDGE:
//			res = get_vertices(std::integral_constant<size_t, EDGE>(), s, v);
//			break;
//		case FACE:
//			res = get_vertices(std::integral_constant<size_t, FACE>(), s, v);
//			break;
//		case VOLUME:
//			res = get_vertices(std::integral_constant<size_t, VOLUME>(), s, v);
//			break;
//		}
//		return res;
//	}
//
//	template<size_t IFORM>
//	static size_t get_vertices(std::integral_constant<size_t, IFORM>, id_type s,
//			id_type *v)
//	{
//		return get_adjacent_cells(std::integral_constant<size_t, IFORM>(),
//				std::integral_constant<size_t, VERTEX>(), s, v);
//	}
	/** @} */
//
//	static constexpr coordinates_type coordinates_local_to_global(id_type s,
//			coordinates_type const &r)
//	{
//		return static_cast<coordinates_type>(id_to_coordinates(s) + r);
//	}
//
//	template<typename TZ>
//	static constexpr coordinates_type coordinates_local_to_global(TZ const &z)
//	{
//		return std::move(
//				coordinates_local_to_global(std::get<0>(z), std::get<1>(z)));
//	}
//
//	/**
//	 *
//	 * @param x coordinates \f$ x \in \left[0,MX\right)\f$
//	 * @param shift
//	 * @return s,r  s is the largest grid point not greater than x.
//	 *       and  \f$ r \in \left[0,1.0\right) \f$ is the normalize  distance between x and s
//	 */
//
//	template<size_t IFORM = 0, size_t N = 0>
//	static id_type coordinates_global_to_local(coordinates_type * x)
//	{
//		index_tuple I;
//
//		*x += m_shift_[IFORM][N];
//
//		x[0] = std::modf(x[0], &I[0]);
//
//		id_type s = (coordinates_to_id(*x) & CELL_ID_MASK);
//
//		coordinates_type r;
//
//		r = x - id_to_coordinates(s);
//
//		return std::move(s);
//	}
//
//	/**
//	 *
//	 * @param x  coordinates \f$ x \in \left[0,1\right)\f$
//	 * @param shift
//	 * @return s,r   s is thte conmpact index of nearest grid point
//	 *    and  \f$ r \in \left[-0.5,0.5\right) \f$   is the normalize  distance between x and s
//	 */
//	static id_type coordinates_global_to_local_NGP(coordinates_type * x)
//	{
//		auto & x = std::get<1>(z);
//		id_type shift = std::get<0>(z);
//
//		index_tuple I = id_to_index(shift >> (FLOATING_POINT_POS - 1));
//
//		coordinates_type r;
//
//		r[0] = x[0] - 0.5 * static_cast<Real>(I[0]);
//		r[1] = x[1] - 0.5 * static_cast<Real>(I[1]);
//		r[2] = x[2] - 0.5 * static_cast<Real>(I[2]);
//
//		I[0] = static_cast<index_type>(std::floor(r[0] + 0.5));
//		I[1] = static_cast<index_type>(std::floor(r[1] + 0.5));
//		I[2] = static_cast<index_type>(std::floor(r[2] + 0.5));
//
//		r -= I;
//
//		id_type s = (index_to_id(I)) | shift;
//
//		return std::move(std::make_tuple(s, r));
//	}
//
//	//! @}
//
//	//! @name id auxiliary functions
//	//! @{
//	static constexpr id_type dual(id_type r)
//	{
//		return (r & (~_DA)) | ((~(r & _DA)) & _DA);
//
//	}
//	static constexpr id_type get_cell_id(id_type r)
//	{
//		return r & CELL_ID_MASK;
//	}
//
//	static constexpr id_type node_id(id_type s)
//	{
//
//		return (((s >> (INDEX_DIGITS * 2 + FLOATING_POINT_POS - 1)) & 1UL) << 2)
//				| (((s >> (INDEX_DIGITS + FLOATING_POINT_POS - 1)) & 1UL) << 1)
//				| ((s >> (FLOATING_POINT_POS - 1)) & 1UL);
//
//	}
//
//	/**
//	 *  rotate vector direction  mask
//	 *  (1/2,0,0) => (0,0,1/2) or   (1/2,1/2,0) => (1/2,0,1/2)
//	 * @param s
//	 * @return
//	 */
//
//	static constexpr id_type delta_index(id_type r)
//	{
//		return (r & _DA);
//	}
//
//	static constexpr id_type DI(size_t i, id_type r)
//	{
//		return (1UL << (INDEX_DIGITS * i + FLOATING_POINT_POS - 1));
//
//	}
//	static constexpr id_type delta_index(size_t i, id_type r)
//	{
//		return DI(i, r) & r;
//	}
//
//	/**
//	 * Get component number or vector direction
//	 * @param s
//	 * @return
//	 */
//
//	static constexpr id_type m_component_number_[] = { 0,  // 000
//			0, // 001
//			1, // 010
//			2, // 011
//			2, // 100
//			1, // 101
//			0, // 110
//			0 };
//
//	static constexpr id_type component_number(id_type s)
//	{
//		return m_component_number_[node_id(s)];
//	}
//
//
//	static constexpr id_type IForm(id_type r)
//	{
//		return m_iform_[node_id(r)];
//	}
//	//! @}
//	/**
//	 * @name Neighgour
//	 * @{
//	 */
//
//	static size_t get_vertices(id_type s, id_type *v)
//	{
//		size_t res = 0;
//		switch (IForm(s))
//		{
//		case VERTEX:
//			res = get_vertices(std::integral_constant<size_t, VERTEX>(), s, v);
//			break;
//		case EDGE:
//			res = get_vertices(std::integral_constant<size_t, EDGE>(), s, v);
//			break;
//		case FACE:
//			res = get_vertices(std::integral_constant<size_t, FACE>(), s, v);
//			break;
//		case VOLUME:
//			res = get_vertices(std::integral_constant<size_t, VOLUME>(), s, v);
//			break;
//		}
//		return res;
//	}
//
//	template<size_t IFORM>
//	static size_t get_vertices(std::integral_constant<size_t, IFORM>, id_type s,
//			id_type *v)
//	{
//		return get_adjacent_cells(std::integral_constant<size_t, IFORM>(),
//				std::integral_constant<size_t, VERTEX>(), s, v);
//	}
//
//	template<size_t I>
//	static inline size_t get_adjacent_cells(std::integral_constant<size_t, I>,
//			std::integral_constant<size_t, I>, id_type s, id_type *v)
//	{
//		v[0] = s;
//		return 1;
//	}
//
//	static inline size_t get_adjacent_cells(
//			std::integral_constant<size_t, EDGE>,
//			std::integral_constant<size_t, VERTEX>, id_type s, id_type *v)
//	{
//		v[0] = s + delta_index(s);
//		v[1] = s - delta_index(s);
//		return 2;
//	}
//
//	static inline size_t get_adjacent_cells(
//			std::integral_constant<size_t, FACE>,
//			std::integral_constant<size_t, VERTEX>, id_type s, id_type *v)
//	{
//		/**
//		 * \verbatim
//		 *                ^y
//		 *               /
//		 *        z     /
//		 *        ^
//		 *        |   2---------------*
//		 *        |  /|              /|
//		 *          / |             / |
//		 *         /  |            /  |
//		 *        3---|-----------*   |
//		 *        | m |           |   |
//		 *        |   1-----------|---*
//		 *        |  /            |  /
//		 *        | /             | /
//		 *        |/              |/
//		 *        0---------------*---> x
//		 * \endverbatim
//		 *
//		 */
//
//		auto di = delta_index(roate(dual(s)));
//		auto dj = delta_index(inverse_roate(dual(s)));
//
//		v[0] = s - di - dj;
//		v[1] = s - di - dj;
//		v[2] = s + di + dj;
//		v[3] = s + di + dj;
//
//		return 4;
//	}
//
//	static inline size_t get_adjacent_cells(
//			std::integral_constant<size_t, VOLUME>,
//			std::integral_constant<size_t, VERTEX>, id_type s, id_type *v)
//	{
//		/**
//		 * \verbatim
//		 *                ^y
//		 *               /
//		 *        z     /
//		 *        ^
//		 *        |   6---------------7
//		 *        |  /|              /|
//		 *          / |             / |
//		 *         /  |            /  |
//		 *        4---|-----------5   |
//		 *        |   |           |   |
//		 *        |   2-----------|---3
//		 *        |  /            |  /
//		 *        | /             | /
//		 *        |/              |/
//		 *        0---------------1   ---> x
//		 *
//		 *   \endverbatim
//		 */
//		auto di = DI(0, s);
//		auto dj = DI(1, s);
//		auto dk = DI(2, s);
//
//		v[0] = ((s - di) - dj) - dk;
//		v[1] = ((s - di) - dj) + dk;
//		v[2] = ((s - di) + dj) - dk;
//		v[3] = ((s - di) + dj) + dk;
//
//		v[4] = ((s + di) - dj) - dk;
//		v[5] = ((s + di) - dj) + dk;
//		v[6] = ((s + di) + dj) - dk;
//		v[7] = ((s + di) + dj) + dk;
//
//		return 8;
//	}
//
//	static inline size_t get_adjacent_cells(
//			std::integral_constant<size_t, VERTEX>,
//			std::integral_constant<size_t, EDGE>, id_type s, id_type *v)
//	{
//		/**
//		 * \verbatim
//		 *
//		 *                ^y
//		 *               /
//		 *        z     /
//		 *        ^
//		 *        |   6---------------7
//		 *        |  /|              /|
//		 *          2 |             / |
//		 *         /  1            /  |
//		 *        4---|-----------5   |
//		 *        |   |           |   |
//		 *        |   2-----------|---3
//		 *        3  /            |  /
//		 *        | 0             | /
//		 *        |/              |/
//		 *        0------E0-------1   ---> x
//		 *
//		 * \endverbatim
//		 */
//
//		auto di = DI(0, s);
//		auto dj = DI(1, s);
//		auto dk = DI(2, s);
//
//		v[0] = s + di;
//		v[1] = s - di;
//
//		v[2] = s + dj;
//		v[3] = s - dj;
//
//		v[4] = s + dk;
//		v[5] = s - dk;
//
//		return 6;
//	}
//
//	static inline size_t get_adjacent_cells(
//			std::integral_constant<size_t, FACE>,
//			std::integral_constant<size_t, EDGE>, id_type s, id_type *v)
//	{
//
//		/**
//		 *\verbatim
//		 *                ^y
//		 *               /
//		 *        z     /
//		 *        ^
//		 *        |   6---------------7
//		 *        |  /|              /|
//		 *          2 |             / |
//		 *         /  1            /  |
//		 *        4---|-----------5   |
//		 *        |   |           |   |
//		 *        |   2-----------|---3
//		 *        3  /            |  /
//		 *        | 0             | /
//		 *        |/              |/
//		 *        0---------------1   ---> x
//		 *
//		 *\endverbatim
//		 */
//		auto d1 = delta_index(roate(dual(s)));
//		auto d2 = delta_index(inverse_roate(dual(s)));
//		v[0] = s - d1;
//		v[1] = s + d1;
//		v[2] = s - d2;
//		v[3] = s + d2;
//
//		return 4;
//	}
//
//	static inline size_t get_adjacent_cells(
//			std::integral_constant<size_t, VOLUME>,
//			std::integral_constant<size_t, EDGE>, id_type s, id_type *v)
//	{
//
//		/**
//		 *\verbatim
//		 *                ^y
//		 *               /
//		 *        z     /
//		 *        ^
//		 *        |   6------10-------7
//		 *        |  /|              /|
//		 *         11 |             9 |
//		 *         /  7            /  6
//		 *        4---|---8-------5   |
//		 *        |   |           |   |
//		 *        |   2-------2---|---3
//		 *        4  /            5  /
//		 *        | 3             | 1
//		 *        |/              |/
//		 *        0-------0-------1   ---> x
//		 *
//		 *\endverbatim
//		 */
//		auto di = DI(0, s);
//		auto dj = DI(1, s);
//		auto dk = DI(2, s);
//
//		v[0] = (s + di) + dj;
//		v[1] = (s + di) - dj;
//		v[2] = (s - di) + dj;
//		v[3] = (s - di) - dj;
//
//		v[4] = (s + dk) + dj;
//		v[5] = (s + dk) - dj;
//		v[6] = (s - dk) + dj;
//		v[7] = (s - dk) - dj;
//
//		v[8] = (s + di) + dk;
//		v[9] = (s + di) - dk;
//		v[10] = (s - di) + dk;
//		v[11] = (s - di) - dk;
//
//		return 12;
//	}
//
//	static inline size_t get_adjacent_cells(
//			std::integral_constant<size_t, VERTEX>,
//			std::integral_constant<size_t, FACE>, id_type s, id_type *v)
//	{
//		/**
//		 *\verbatim
//		 *                ^y
//		 *               /
//		 *        z     /
//		 *        ^
//		 *        |   6---------------7
//		 *        |  /|              /|
//		 *        | / |             / |
//		 *        |/  |            /  |
//		 *        4---|-----------5   |
//		 *        |   |           |   |
//		 *        | 0 2-----------|---3
//		 *        |  /            |  /
//		 *   11   | /      8      | /
//		 *      3 |/              |/
//		 * -------0---------------1   ---> x
//		 *       /| 1
//		 *10    / |     9
//		 *     /  |
//		 *      2 |
//		 *
//		 *
//		 *
//		 *              |
//		 *          7   |   4
//		 *              |
//		 *      --------*---------
//		 *              |
//		 *          6   |   5
//		 *              |
//		 *
//		 *\endverbatim
//		 */
//		auto di = DI(0, s);
//		auto dj = DI(1, s);
//		auto dk = DI(2, s);
//
//		v[0] = (s + di) + dj;
//		v[1] = (s + di) - dj;
//		v[2] = (s - di) + dj;
//		v[3] = (s - di) - dj;
//
//		v[4] = (s + dk) + dj;
//		v[5] = (s + dk) - dj;
//		v[6] = (s - dk) + dj;
//		v[7] = (s - dk) - dj;
//
//		v[8] = (s + di) + dk;
//		v[9] = (s + di) - dk;
//		v[10] = (s - di) + dk;
//		v[11] = (s - di) - dk;
//
//		return 12;
//	}
//
//	static inline size_t get_adjacent_cells(
//			std::integral_constant<size_t, EDGE>,
//			std::integral_constant<size_t, FACE>, id_type s, id_type *v)
//	{
//
//		/**
//		 *\verbatim
//		 *                ^y
//		 *               /
//		 *        z     /
//		 *        ^
//		 *        |   6---------------7
//		 *        |  /|              /|
//		 *        | / |             / |
//		 *        |/  |            /  |
//		 *        4---|-----------5   |
//		 *        |   |           |   |
//		 *        |   2-----------|---3
//		 *        |  /  0         |  /
//		 *        | /      1      | /
//		 *        |/              |/
//		 * -------0---------------1   ---> x
//		 *       /|
//		 *      / |   3
//		 *     /  |       2
//		 *        |
//		 *
//		 *
//		 *
//		 *              |
//		 *          7   |   4
//		 *              |
//		 *      --------*---------
//		 *              |
//		 *          6   |   5
//		 *              |
//		 *
//		 *\endverbatim
//		 */
//
//		auto d1 = delta_index(roate((s)));
//		auto d2 = delta_index(inverse_roate((s)));
//
//		v[0] = s - d1;
//		v[1] = s + d1;
//		v[2] = s - d2;
//		v[3] = s + d2;
//
//		return 4;
//	}
//
//	static inline size_t get_adjacent_cells(
//			std::integral_constant<size_t, VOLUME>,
//			std::integral_constant<size_t, FACE>, id_type s, id_type *v)
//	{
//
//		/**
//		 *\verbatim
//		 *                ^y
//		 *               /
//		 *        z     /
//		 *        ^    /
//		 *        |   6---------------7
//		 *        |  /|              /|
//		 *        | / |    5        / |
//		 *        |/  |     1      /  |
//		 *        4---|-----------5   |
//		 *        | 0 |           | 2 |
//		 *        |   2-----------|---3
//		 *        |  /    3       |  /
//		 *        | /       4     | /
//		 *        |/              |/
//		 * -------0---------------1   ---> x
//		 *       /|
//		 *\endverbatim
//		 */
//
//		auto di = DI(0, s);
//		auto dj = DI(1, s);
//		auto dk = DI(2, s);
//
//		v[0] = s - di;
//		v[1] = s + di;
//
//		v[2] = s - di;
//		v[3] = s + dj;
//
//		v[4] = s - dk;
//		v[5] = s + dk;
//
//		return 6;
//	}
//
//	static inline size_t get_adjacent_cells(
//			std::integral_constant<size_t, VERTEX>,
//			std::integral_constant<size_t, VOLUME>, id_type s, id_type *v)
//	{
//		/**
//		 *\verbatim
//		 *                ^y
//		 *               /
//		 *        z     /
//		 *        ^
//		 *        |   6---------------7
//		 *        |  /|              /|
//		 *        | / |             / |
//		 *        |/  |            /  |
//		 *        4---|-----------5   |
//		 *   3    |   |    0      |   |
//		 *        |   2-----------|---3
//		 *        |  /            |  /
//		 *        | /             | /
//		 *        |/              |/
//		 * -------0---------------1   ---> x
//		 *  3    /|       1
//		 *      / |
//		 *     /  |
//		 *        |
//		 *
//		 *
//		 *
//		 *              |
//		 *          7   |   4
//		 *              |
//		 *      --------*---------
//		 *              |
//		 *          6   |   5
//		 *              |
//		 *
//		 *\endverbatim
//		 */
//
//		auto di = DI(0, s);
//		auto dj = DI(1, s);
//		auto dk = DI(2, s);
//
//		v[0] = ((s - di) - dj) - dk;
//		v[1] = ((s - di) - dj) + dk;
//		v[2] = ((s - di) + dj) - dk;
//		v[3] = ((s - di) + dj) + dk;
//
//		v[4] = ((s + di) - dj) - dk;
//		v[5] = ((s + di) - dj) + dk;
//		v[6] = ((s + di) + dj) - dk;
//		v[7] = ((s + di) + dj) + dk;
//
//		return 8;
//	}
//
//	static inline size_t get_adjacent_cells(
//			std::integral_constant<size_t, EDGE>,
//			std::integral_constant<size_t, VOLUME>, id_type s, id_type *v)
//	{
//
//		/**
//		 *\verbatim
//		 *                ^y
//		 *               /
//		 *        z     /
//		 *        ^
//		 *        |   6---------------7
//		 *        |  /|              /|
//		 *        | / |             / |
//		 *        |/  |            /  |
//		 *        4---|-----------5   |
//		 *        |   |           |   |
//		 *        |   2-----------|---3
//		 *        |  /  0         |  /
//		 *        | /      1      | /
//		 *        |/              |/
//		 * -------0---------------1   ---> x
//		 *       /|
//		 *      / |   3
//		 *     /  |       2
//		 *        |
//		 *
//		 *
//		 *
//		 *              |
//		 *          7   |   4
//		 *              |
//		 *      --------*---------
//		 *              |
//		 *          6   |   5
//		 *              |
//		 *
//		 *\endverbatim
//		 */
//
//		auto d1 = delta_index(roate((s)));
//		auto d2 = delta_index(inverse_roate((s)));
//
//		v[0] = s - d1 - d2;
//		v[1] = s + d1 - d2;
//		v[2] = s - d1 + d2;
//		v[3] = s + d1 + d2;
//		return 4;
//	}
//
//	static inline size_t get_adjacent_cells(
//			std::integral_constant<size_t, FACE>,
//			std::integral_constant<size_t, VOLUME>, id_type s, id_type *v)
//	{
//
//		/**
//		 *\verbatim
//		 *                ^y
//		 *               /
//		 *        z     /
//		 *        ^    /
//		 *        |   6---------------7
//		 *        |  /|              /|
//		 *        | / |             / |
//		 *        |/  |            /  |
//		 *        4---|-----------5   |
//		 *        | 0 |           |   |
//		 *        |   2-----------|---3
//		 *        |  /            |  /
//		 *        | /             | /
//		 *        |/              |/
//		 * -------0---------------1   ---> x
//		 *       /|
//		 *\endverbatim
//		 */
//
//		auto d = delta_index(dual(s));
//		v[0] = s + d;
//		v[1] = s - d;
//
//		return 2;
//	}
//	/**@}*/
}
;
template<size_t N, size_t A> constexpr size_t MeshIDs_<N, A>::m_sub_node_id_[4][3];
/**
 * Solve problem: Undefined reference to static constexpr char[]
 * http://stackoverflow.com/questions/22172789/passing-a-static-constexpr-variable-by-universal-reference
 */

template<size_t N, size_t A> constexpr size_t MeshIDs_<N, A>::FLOATING_POINT_FACTOR;
template<size_t N, size_t A> constexpr size_t MeshIDs_<N, A>::FULL_DIGITS;
template<size_t N, size_t A> constexpr size_t MeshIDs_<N, A>::INDEX_DIGITS;
template<size_t N, size_t A> constexpr size_t MeshIDs_<N, A>::FLOATING_POINT_POS;
template<size_t N, size_t A> constexpr size_t MeshIDs_<N, A>::INDEX_MASK;
template<size_t N, size_t A> constexpr size_t MeshIDs_<N, A>::D_INDEX;
template<size_t N, size_t A> constexpr size_t MeshIDs_<N, A>::_DK;
template<size_t N, size_t A> constexpr size_t MeshIDs_<N, A>::_DJ;
template<size_t N, size_t A> constexpr size_t MeshIDs_<N, A>::_DI;
template<size_t N, size_t A> constexpr size_t MeshIDs_<N, A>::_DA;
template<size_t N, size_t A> constexpr size_t MeshIDs_<N, A>::CELL_ID_MASK_;
template<size_t N, size_t A> constexpr size_t MeshIDs_<N, A>::CELL_ID_MASK;
template<size_t N, size_t A> constexpr size_t MeshIDs_<N, A>::ID_ZERO;
typedef MeshIDs_<3, 0> MeshIDs;
}  // namespace simpla

#endif /* CORE_MESH_MESH_IDS_H_ */
