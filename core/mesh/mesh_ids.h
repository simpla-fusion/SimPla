/**
 * @file mesh_ids.h
 *
 * @date 2015年3月19日
 * @author salmon
 */

#ifndef CORE_MESH_MESH_IDS_H_
#define CORE_MESH_MESH_IDS_H_

#include "../gtl/containers/sp_hash_container.h"
#include "../gtl/iterator/sp_ntuple_range.h"
#include "../gtl/ntuple.h"
#include "../gtl/primitives.h"
#include "../gtl/type_traits.h"
//
#include "../parallel/mpi_comm.h"
#include "../parallel/mpi_aux_functions.h"
#include <stddef.h>
#include <algorithm>
#include <limits>
#include <cmath>

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
template<size_t NDIMS = 3, size_t INIFIT_AXIS = 0>
struct MeshIDs_
{
	typedef size_t id_type;

//	typedef typename std::make_signed<id_type>::type index_type;
	typedef long index_type;

	typedef nTuple<Real, 3> coordinates_type;

	typedef nTuple<index_type, 3> index_tuple;

	static constexpr int ndims = NDIMS;

	static constexpr size_t FULL_DIGITS = std::numeric_limits<size_t>::digits;

	static constexpr size_t INDEX_DIGITS = (FULL_DIGITS) / 3;

	static constexpr size_t INDEX_MASK = (1UL << (INDEX_DIGITS)) - 1;

	static constexpr size_t MAX_MESH_LEVEL = 4;

	static constexpr size_t _DI = (1UL << (MAX_MESH_LEVEL - 1));

	static constexpr size_t _DJ = _DI << (INDEX_DIGITS);

	static constexpr size_t _DK = _DI << (INDEX_DIGITS * 2);

	static constexpr size_t _DA = _DI | _DJ | _DK;

	static constexpr index_type INDEX_ZERO = static_cast<index_type>(1UL
			<< (INDEX_DIGITS - MAX_MESH_LEVEL - 1));

	static constexpr Real COORD_ZERO = static_cast<Real>(INDEX_ZERO);

	static constexpr Real COORD_TO_INDEX_FACTOR = static_cast<Real>(1
			<< (MAX_MESH_LEVEL));

	static constexpr Real INDEX_TO_COORD_FACTOR = 1.0 / COORD_TO_INDEX_FACTOR;

//	static constexpr size_t CELL_ID_MASK_ = //
//			(((1UL << (ID_DIGITS - MAX_NUM_OF_MESH_LEVEL)) - 1)
//					<< (MAX_NUM_OF_MESH_LEVEL)) & ID_MASK;
//
//	static constexpr size_t CELL_ID_MASK =
//
//	(CELL_ID_MASK_ << (ID_DIGITS * 2))
//
//	| (CELL_ID_MASK_ << (ID_DIGITS))
//
//	| (CELL_ID_MASK_);

//	static constexpr size_t CELL_ID_MASK =
//
//	(((INIFIT_AXIS & 1UL) == 0) ? (INDEX_MASK) : 0UL)
//
//	| (((INIFIT_AXIS & 2UL) == 0) ? (INDEX_MASK << INDEX_DIGITS) : 0UL)
//
//	| (((INIFIT_AXIS & 4UL) == 0) ? (INDEX_MASK << (INDEX_DIGITS * 2)) : 0UL);

	static constexpr size_t m_sub_node_num_[4][3] = { //

			{ 0, 0, 0 }, /*VERTEX*/
			{ 1, 2, 4 }, /*EDGE*/
			{ 6, 5, 3 }, /*FACE*/
			{ 7, 7, 7 } /*VOLUME*/

			};

	static constexpr size_t m_sub_node_id_shift_[4][3] = {

	{ 0, 0, 0 },

	{ _DI, _DJ, _DK },

	{ (_DJ | _DK), (_DK | _DI), (_DI | _DJ) },

	{ _DA, _DA, _DA }

	};

	static constexpr coordinates_type m_sub_node_coordinates_shift_[4][3] = {

	{ { 0, 0, 0 }, { 0, 0, 0 }, { 0, 0, 0 } }, /*VERTEX*/
	{ { 0.5, 0, 0 }, { 0, 0.5, 0 }, { 0, 0, 0.5 } }, /*EDGE*/
	{ { 0, 0.5, 0.5 }, { 0.5, 0, 0.5 }, { 0.5, 0.5, 0 } }, /*FACE*/
	{ { 0, 0.5, 0.5 }, { 0.5, 0.5, 0.5 }, { 0.5, 0.5, 0.5 } } /*VOLUME*/

	};

	static constexpr id_type m_shift_to_iform_[] = { //

			VERTEX, // 000
					EDGE, // 001
					EDGE, // 010
					FACE, // 011
					EDGE, // 100
					FACE, // 101
					FACE, // 110
					VOLUME // 111
			};

	template<int IFORM, typename ...Args>
	static id_type id(Args &&...args)
	{
		return pack<IFORM>(std::forward<Args>(args)...);
	}

	template<size_t IFORM, typename TInt>
	static constexpr id_type pack(TInt i, TInt j, TInt k, int n = 0)
	{
		return

		m_sub_node_id_shift_[IFORM][n % 3]

		| (static_cast<size_t>(i + INDEX_ZERO) << MAX_MESH_LEVEL)

				| (static_cast<size_t>(j + INDEX_ZERO)
						<< (MAX_MESH_LEVEL + INDEX_DIGITS))

				| (static_cast<size_t>(k + INDEX_ZERO)

				<< (MAX_MESH_LEVEL + INDEX_DIGITS * 2));
	}

	template<size_t IFORM, typename TInt>
	static constexpr id_type pack(nTuple<TInt, 3> const &i, int n = 0)
	{
		return pack<IFORM>(i[0], i[1], i[2], n);
	}

	template<size_t IFORM, typename TInt>
	static constexpr id_type pack(nTuple<TInt, 4> const &i)
	{
		return pack<IFORM>(i[0], i[1], i[2], i[3]);
	}

	template<size_t IFORM>
	static nTuple<index_type, NDIMS> unpack(id_type s)
	{
		return nTuple<index_type, NDIMS>(
				{ static_cast<index_type>((s & INDEX_MASK) >> (MAX_MESH_LEVEL))
						- INDEX_ZERO,

				static_cast<index_type>(((s >> (INDEX_DIGITS)) & INDEX_MASK)
						>> (MAX_MESH_LEVEL)) - INDEX_ZERO,

				static_cast<index_type>(((s >> (INDEX_DIGITS * 2)) & INDEX_MASK)
						>> (MAX_MESH_LEVEL)) - INDEX_ZERO })

		;
	}

	template<size_t MESH_LEVEL = 0>
	static index_tuple id_to_index(id_type const &s)
	{
		return std::move(
				index_tuple(
						{

						static_cast<long>((s & INDEX_MASK)
								>> (MAX_MESH_LEVEL - MESH_LEVEL))
								- static_cast<index_type>(INDEX_ZERO
										<< MESH_LEVEL),

						static_cast<long>(((s >> (INDEX_DIGITS)) & INDEX_MASK)
								>> (MAX_MESH_LEVEL - MESH_LEVEL))
								- static_cast<index_type>(INDEX_ZERO
										<< MESH_LEVEL),

						static_cast<long>(((s >> (INDEX_DIGITS * 2))
								& INDEX_MASK) >> (MAX_MESH_LEVEL - MESH_LEVEL))
								- static_cast<index_type>(INDEX_ZERO
										<< MESH_LEVEL)

						}));
	}

	template<size_t IFORM, typename TX>
	static constexpr id_type coordinates_to_id(TX const &x, int n = 0)
	{

		return m_sub_node_id_shift_[IFORM][n]

				| (static_cast<size_t>(std::floor(
						(x[0] - m_sub_node_coordinates_shift_[IFORM][n][0]
								+ COORD_ZERO) * COORD_TO_INDEX_FACTOR) + 0.5)
						& INDEX_MASK)

				| ((static_cast<size_t>(std::floor(
						(x[1] - m_sub_node_coordinates_shift_[IFORM][n][1]
								+ COORD_ZERO) * COORD_TO_INDEX_FACTOR) + 0.5)
						& INDEX_MASK) << (INDEX_DIGITS))

				| ((static_cast<size_t>(std::floor(

						(x[2] - m_sub_node_coordinates_shift_[IFORM][n][2]
								+ COORD_ZERO) * COORD_TO_INDEX_FACTOR) + 0.5)
						& INDEX_MASK) << (INDEX_DIGITS * 2))

		;
	}
	static constexpr coordinates_type id_to_coordinates(id_type s)
	{

		return std::move(coordinates_type { //

				static_cast<Real>((s & INDEX_MASK)) * INDEX_TO_COORD_FACTOR
						- COORD_ZERO,

				static_cast<Real>(((s >> INDEX_DIGITS)) & INDEX_MASK)
						* INDEX_TO_COORD_FACTOR - COORD_ZERO,

				static_cast<Real>(((s >> (INDEX_DIGITS * 2))) & INDEX_MASK)
						* INDEX_TO_COORD_FACTOR - COORD_ZERO

				}

				);
	}
	template<size_t IFORM, typename TX> static constexpr index_tuple coordinates_to_index(
			TX const &x, int n = 0)
	{
		return
		{

			(static_cast<index_type>(std::floor(x[0] - m_sub_node_coordinates_shift_[IFORM][n][0])/**COORD_TO_INDEX_FACTOR*/)),

			(static_cast<index_type>(std::floor(x[1] - m_sub_node_coordinates_shift_[IFORM][n][1])/**COORD_TO_INDEX_FACTOR*/)),

			(static_cast<index_type>(std::floor(x[2] - m_sub_node_coordinates_shift_[IFORM][n][2])/**COORD_TO_INDEX_FACTOR*/))

		};
	}

	template<size_t IFORM, typename TX> static std::tuple<id_type,
			coordinates_type> coordinates_global_to_local(TX const &y,
			int n = 0)
	{
		nTuple<int, 3> idx;

		coordinates_type x;

		x = (y - m_sub_node_coordinates_shift_[IFORM][n])
				* COORD_TO_INDEX_FACTOR;

		x[0] = std::remquo(x[0], COORD_TO_INDEX_FACTOR, &idx[0])
				* INDEX_TO_COORD_FACTOR;
		x[1] = std::remquo(x[1], COORD_TO_INDEX_FACTOR, &idx[1])
				* INDEX_TO_COORD_FACTOR;
		x[2] = std::remquo(x[2], COORD_TO_INDEX_FACTOR, &idx[2])
				* INDEX_TO_COORD_FACTOR;

		return std::make_tuple(id<IFORM>(idx, n), x);

	}

//! @name id auxiliary functions
//! @{
	static constexpr id_type dual(id_type s)
	{
		return (s & (~_DA)) | ((~(s & _DA)) & _DA);

	}

	static constexpr id_type delta_index(id_type s)
	{
		return (s & _DA);
	}

	static constexpr id_type roate(id_type const &s)
	{
		return ((s & (_DA)) >> INDEX_DIGITS) | ((s & _DI) << (INDEX_DIGITS * 2));
	}

	static constexpr id_type inverse_roate(id_type const &s)
	{
		return ((s & (_DA)) << INDEX_DIGITS) | ((s & _DK) >> (INDEX_DIGITS * 2));
	}

	static constexpr int m_node_id_[8] = { 0, // 000
			0, // 001
			1, // 010
			2, // 011
			2, // 100
			1, // 101
			0, // 110
			0, // 111
			};

	static constexpr int node_id(id_type const &s)
	{
		return m_node_id_[(((s & _DI) >> (MAX_MESH_LEVEL - 1))
				| ((s & _DJ) >> (INDEX_DIGITS + MAX_MESH_LEVEL - 2))
				| ((s & _DK) >> (INDEX_DIGITS * 2 + MAX_MESH_LEVEL - 3))) & 7UL];
	}

//	typedef SpHashContainer<id_type, Real, id_hasher<1>> volume_container;
//
//	template<typename ...Args>
//	static volume_container make_volume_container(Args &&...args)
//	{
//		id_hasher<1> hasher(std::forward<Args>(args)...);
//
//		return std::move(volume_container((hasher), hasher.max_hash()));
//	}

	template<size_t IFORM> static constexpr int get_vertics(int n, id_type s,
			coordinates_type *q = nullptr)
	{
		return get_vertics_(std::integral_constant<size_t, IFORM>(), n, s, q);
	}

	/**
	 * \verbatim
	 *                ^y
	 *               /
	 *        z     /
	 *        ^
	 *        |  q7--------------q6
	 *        |  /|              /|
	 *          / |             / |
	 *         /  |            /  |
	 *       q4---|----------q5   |
	 *        |   |     x0    |   |
	 *        |  q3-----------|--q2
	 *        |  /            |  /
	 *        | /             | /
	 *        |/              |/
	 *       q0--------------q1   ---> x
	 *
	 *   \endverbatim
	 */

	static int get_vertics_(std::integral_constant<size_t, VOLUME>, int n,
			id_type s, coordinates_type *q = nullptr)
	{

		if (q != nullptr)
		{
			coordinates_type x0 = id_to_coordinates(s);

			coordinates_type dx = id_to_coordinates(_DI | INDEX_ZERO);
			coordinates_type dy = id_to_coordinates(_DJ | INDEX_ZERO);
			coordinates_type dz = id_to_coordinates(_DK | INDEX_ZERO);

			q[0] = x0 - dx - dy - dz;
			q[1] = x0 + dx - dy - dz;
			q[2] = x0 + dx + dy - dz;
			q[3] = x0 - dx + dy - dz;

			q[4] = x0 - dx - dy + dz;
			q[5] = x0 + dx - dy + dz;
			q[6] = x0 + dx + dy + dz;
			q[7] = x0 - dx + dy + dz;
		}

		return 8;
	}

	static int get_vertics_(std::integral_constant<size_t, FACE>, int n,
			id_type s, coordinates_type *q = nullptr)
	{

		if (q != nullptr)
		{
			coordinates_type x0 = id_to_coordinates(s);

			coordinates_type d[3] = {

			id_to_coordinates(_DI | INDEX_ZERO),

			id_to_coordinates(_DJ | INDEX_ZERO),

			id_to_coordinates(_DK | INDEX_ZERO) };

			coordinates_type const &dx = d[(n + 1) % 3];
			coordinates_type const &dy = d[(n + 2) % 3];
			q[0] = x0 - dx - dy;
			q[1] = x0 + dx - dy;
			q[2] = x0 + dx + dy;
			q[3] = x0 - dx + dy;
		}

		return 4;
	}

	static int get_vertics_(std::integral_constant<size_t, EDGE>, int n,
			id_type s, coordinates_type *q = nullptr)
	{

		if (q != nullptr)
		{
			coordinates_type x0 = id_to_coordinates(s);

			coordinates_type d[3] = {

			id_to_coordinates(_DI | INDEX_ZERO),

			id_to_coordinates(_DJ | INDEX_ZERO),

			id_to_coordinates(_DK | INDEX_ZERO) };

			coordinates_type const &dx = d[n];

			q[0] = x0 - dx;
			q[1] = x0 + dx;

		}

		return 4;
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
};

/**
 * Solve problem: Undefined reference to static constexpr char[]
 * http://stackoverflow.com/questions/22172789/passing-a-static-constexpr-variable-by-universal-reference
 */

template<size_t N, size_t A> constexpr size_t MeshIDs_<N, A>::FULL_DIGITS;
template<size_t N, size_t A> constexpr size_t MeshIDs_<N, A>::INDEX_DIGITS;
template<size_t N, size_t A> constexpr size_t MeshIDs_<N, A>::MAX_MESH_LEVEL;
template<size_t N, size_t A> constexpr size_t MeshIDs_<N, A>::INDEX_MASK;
template<size_t N, size_t A> constexpr size_t MeshIDs_<N, A>::_DK;
template<size_t N, size_t A> constexpr size_t MeshIDs_<N, A>::_DJ;
template<size_t N, size_t A> constexpr size_t MeshIDs_<N, A>::_DI;
template<size_t N, size_t A> constexpr size_t MeshIDs_<N, A>::_DA;
template<size_t N, size_t A> constexpr typename MeshIDs_<N,A>::index_type MeshIDs_<N, A>::INDEX_ZERO;
template<size_t N, size_t A> constexpr Real MeshIDs_<N, A>::COORD_ZERO;
template<size_t N, size_t A> constexpr Real MeshIDs_<N, A>::COORD_TO_INDEX_FACTOR;
template<size_t N, size_t A> constexpr Real MeshIDs_<N, A>::INDEX_TO_COORD_FACTOR;

template<size_t N, size_t A> constexpr int MeshIDs_<N, A>::m_node_id_[];
template<size_t N, size_t A> constexpr size_t MeshIDs_<N, A>::m_sub_node_id_shift_[4][3];
template<size_t N, size_t A> constexpr size_t MeshIDs_<N, A>::m_sub_node_num_[4][3];
template<size_t N, size_t A> constexpr typename MeshIDs_<N,A>::coordinates_type
MeshIDs_<N,A>::m_sub_node_coordinates_shift_[4][3];

typedef MeshIDs_<3, 0> MeshIDs;

}
// namespace simpla

#endif /* CORE_MESH_MESH_IDS_H_ */

