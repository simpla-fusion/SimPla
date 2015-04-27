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
template<size_t NDIMS = 3, size_t IMESH_LEVEL = 4, size_t INIFIT_AXIS = 0>
struct MeshIDs_
{
	/// @name level independent
	/// @{

	typedef size_t id_type;

	typedef nTuple<id_type, 3> id_tuple;

	typedef long index_type;

	typedef nTuple<index_type, 3> index_tuple;

	typedef nTuple<Real, 3> coordinates_type;

	static constexpr int ndims = NDIMS;

	static constexpr size_t FULL_DIGITS = std::numeric_limits<size_t>::digits;

	static constexpr size_t ID_DIGITS = (FULL_DIGITS) / 3;

	static constexpr size_t ID_MASK = (1UL << (ID_DIGITS)) - 1;

	static constexpr size_t CARRAY_FLAG = (((1UL) << (ID_DIGITS - 1))
			| (1UL << (ID_DIGITS * 2 - 1)) | (1UL << (ID_DIGITS * 3 - 1)));

	static constexpr size_t CLEAR_CARRAY_FLAG = ~CARRAY_FLAG;

	static constexpr size_t ID_ZERO = (1L << (ID_DIGITS - 2));

	static constexpr Real COORD_ZERO = static_cast<Real>(ID_ZERO);

	/// @}
	/// @name level dependent
	/// @{

	static constexpr int MESH_LEVEL = IMESH_LEVEL;

	static constexpr size_t SUB_ID_MASK = ((1UL << MESH_LEVEL) - 1);

	static constexpr size_t PRIMARY_ID_MASK = ID_MASK & (~SUB_ID_MASK);

	static constexpr size_t _D = (1UL << (MESH_LEVEL - 1));

	static constexpr Real _R = static_cast<Real>(_D);

	static constexpr size_t _DI = _D;

	static constexpr size_t _DJ = 1UL << (ID_DIGITS + MESH_LEVEL - 1);

	static constexpr size_t _DK = 1UL << (ID_DIGITS * 2 + MESH_LEVEL - 1);

	static constexpr size_t _DA = _DI | _DJ | _DK;

	static constexpr index_type INDEX_ZERO = static_cast<index_type>(1L
			<< (ID_DIGITS - MESH_LEVEL - 2));

	static constexpr Real COORDINATES_MESH_FACTOR = static_cast<Real>(1UL
			<< (MESH_LEVEL - 1));

	/// @}

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

	static constexpr size_t m_index_to_node_id_[4][3] =
	{ //

			{ 0, 0, 0 }, /*VERTEX*/
			{ 1, 2, 4 }, /*EDGE*/
			{ 6, 5, 3 }, /*FACE*/
			{ 7, 7, 7 } /*VOLUME*/

			};

	static constexpr size_t m_index_to_id_shift_[4][3] =
	{

	{ 0, 0, 0 },

	{ _DI, _DJ, _DK },

	{ (_DJ | _DK), (_DK | _DI), (_DI | _DJ) },

	{ _DA, _DA, _DA }

	};

	static constexpr coordinates_type m_index_to_coordinates_shift_[4][3] =
	{

	{
	{ 0, 0, 0 },
	{ 0, 0, 0 },
	{ 0, 0, 0 } }, /*VERTEX*/
	{
	{ _R, 0, 0 },
	{ 0, _R, 0 },
	{ 0, 0, _R } }, /*EDGE*/
	{
	{ 0, _R, _R },
	{ _R, 0, _R },
	{ _R, _R, 0 } }, /*FACE*/
	{
	{ 0, _R, _R },
	{ _R, _R, _R },
	{ _R, _R, _R } } /*VOLUME*/

	};

	static constexpr id_type m_node_id_to_iform_[] =
	{ //

			VERTEX, // 000
					EDGE, // 001
					EDGE, // 010
					FACE, // 011
					EDGE, // 100
					FACE, // 101
					FACE, // 110
					VOLUME // 111
			};

	template<size_t IFORM>
	static constexpr id_type pack(id_type i, id_type j, id_type k, int n = 0)
	{
		return

		(m_index_to_id_shift_[IFORM][n % 3])

		| (i)

		| (j << (ID_DIGITS))

		| (k << (ID_DIGITS * 2));
	}

	template<size_t IFORM>
	static constexpr id_type pack(id_tuple const &i, int n = 0)
	{
		return pack<IFORM>(i[0], i[1], i[2], n);
	}
	template<size_t IFORM>
	static constexpr id_type pack(nTuple<id_type, 4> const &i)
	{
		return pack<IFORM>(i[0], i[1], i[2], i[3]);
	}

	template<size_t IFORM>
	static constexpr id_tuple unpack(id_type s)
	{
		return id_tuple(
		{

		((s & ID_MASK)),

		((((s >> (ID_DIGITS)) & ID_MASK))),

		((((s >> (ID_DIGITS * 2)) & ID_MASK)))

		})

		;
	}

	template<size_t IFORM>
	static constexpr id_type pack_index(index_type i, index_type j,
			index_type k, int n = 0)
	{
		return pack<IFORM>(

		static_cast<id_type>(i + INDEX_ZERO) << MESH_LEVEL,

		static_cast<id_type>(j + INDEX_ZERO) << MESH_LEVEL,

		static_cast<id_type>(k + INDEX_ZERO) << MESH_LEVEL,

		n);
	}
	template<size_t IFORM>
	static constexpr id_type pack_index(index_tuple const & idx, int n = 0)
	{
		return pack_index<IFORM>(idx[0], idx[1], idx[2], n);
	}

	template<size_t IFORM>
	static constexpr id_type pack_index(nTuple<index_type, 4> const & idx)
	{
		return pack_index<IFORM>(idx[0], idx[1], idx[2], idx[3]);
	}

	template<size_t IFORM>
	static constexpr index_tuple unpack_index(id_type s)
	{
		return index_tuple(
				{

				static_cast<index_type>((s & ID_MASK) >> MESH_LEVEL)
						- INDEX_ZERO,

				static_cast<index_type>((((s >> (ID_DIGITS)) & ID_MASK))
						>> MESH_LEVEL) - INDEX_ZERO,

				static_cast<index_type>((((s >> (ID_DIGITS * 2)) & ID_MASK))
						>> MESH_LEVEL) - INDEX_ZERO

				});
	}
	template<size_t IFORM>
	static constexpr nTuple<index_type, 4> unpack_index4(id_type s)
	{
		return nTuple<index_type, 4>(
				{

				static_cast<index_type>((s & ID_MASK) >> MESH_LEVEL)
						- INDEX_ZERO,

				static_cast<index_type>((((s >> (ID_DIGITS)) & ID_MASK))
						>> MESH_LEVEL) - INDEX_ZERO,

				static_cast<index_type>((((s >> (ID_DIGITS * 2)) & ID_MASK))
						>> MESH_LEVEL) - INDEX_ZERO,

				sub_index(s)

				});
	}

	template<size_t IFORM = VERTEX>
	static constexpr id_tuple coordinates_to_id_tuple(coordinates_type const &x,
			int n = 0)
	{
		return id_tuple(
				{

				static_cast<id_type>(std::floor(
						x[0] + COORD_ZERO
								- m_index_to_coordinates_shift_[IFORM][n][0]))
						& PRIMARY_ID_MASK,

				static_cast<id_type>(std::floor(
						x[1] + COORD_ZERO
								- m_index_to_coordinates_shift_[IFORM][n][1]))
						& PRIMARY_ID_MASK,

				static_cast<id_type>(std::floor(
						x[2] + COORD_ZERO
								- m_index_to_coordinates_shift_[IFORM][n][2]))
						& PRIMARY_ID_MASK

				})

		;
	}

	template<size_t IFORM = VERTEX>
	static constexpr coordinates_type id_to_coordinates(id_tuple const &idx,
			int n = 0)
	{

		return coordinates_type(
				{

				static_cast<Real>(idx[0])
						+ m_index_to_coordinates_shift_[IFORM][n][0] - ID_ZERO,

				static_cast<Real>(idx[1])
						+ m_index_to_coordinates_shift_[IFORM][n][1] - ID_ZERO,

				static_cast<Real>(idx[2])
						+ m_index_to_coordinates_shift_[IFORM][n][2] - ID_ZERO

				});
	}

	template<size_t IFORM = VERTEX>
	static constexpr id_type coordinates_to_id(coordinates_type const &x,
			int n = 0)
	{
		return pack<IFORM>(coordinates_to_id_tuple<IFORM>(x, n), n);
	}

	template<size_t IFORM = VERTEX>
	static constexpr coordinates_type id_to_coordinates(id_type s)
	{
		return id_to_coordinates<IFORM>(unpack<IFORM>(s), sub_index(s));
	}
	template<size_t IFORM = VERTEX>
	static constexpr coordinates_type coordinates(id_type s)
	{
		return std::move(id_to_coordinates<IFORM>(s));
	}

	template<size_t IFORM = VERTEX>
	static constexpr index_tuple coordinates_to_index(
			coordinates_type const & x, int n = 0)
	{
		return unpack_index<IFORM>(coordinates_to_id<IFORM>(x, n));
	}

	template<size_t IFORM, typename TX>
	static std::tuple<id_type, coordinates_type> coordinates_global_to_local(
			TX const &x, int n = 0)
	{
		id_tuple idx = coordinates_to_id_tuple(x, n);

		coordinates_type r;

		r = (x + COORD_ZERO - idx) / (_R * 2.0);

		return std::make_tuple(pack<IFORM>(idx, n), r);

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

	static constexpr id_type rotate(id_type const &s)
	{
		return ((s & (_DA)) << ID_DIGITS) | ((s & _DK) >> (ID_DIGITS * 2));
	}

	static constexpr id_type inverse_rotate(id_type const &s)
	{
		return ((s & (_DA)) >> ID_DIGITS) | ((s & _DI) << (ID_DIGITS * 2));
	}

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
	static constexpr int node_id(id_type const &s)
	{
		return ((s >> (MESH_LEVEL - 1)) & 1UL)
				| ((s >> (ID_DIGITS + MESH_LEVEL - 2)) & 2UL)
				| ((s >> (ID_DIGITS * 2 + MESH_LEVEL - 3)) & 4UL);
	}

	static constexpr int m_node_id_to_index_[8] =
	{ //

			0, // 000
					0, // 001
					1, // 010
					2, // 011
					2, // 100
					1, // 101
					0, // 110
					0, // 111
			};

	static constexpr int sub_index(id_type const &s)
	{
		return m_node_id_to_index_[node_id(s)];
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
	static constexpr int MAX_NUM_OF_CELL = 12;

	static constexpr size_t _HI = _D;
	static constexpr size_t _HJ = _HI << ID_DIGITS;
	static constexpr size_t _HK = _HI << (ID_DIGITS * 2);
	static constexpr size_t _LI = (((-_D) & ID_MASK) & CLEAR_CARRAY_FLAG);
	static constexpr size_t _LJ = _LI << ID_DIGITS;
	static constexpr size_t _LK = _LI << (ID_DIGITS * 2);

	static constexpr int m_vertics_num_[4/* to iform*/][8/* node id*/] =

	{ // VERTEX
			{
			/* 000*/1,
			/* 001*/2,
			/* 010*/2,
			/* 011*/4,
			/* 100*/2,
			/* 101*/4,
			/* 110*/4,
			/* 111*/8 },

			// EDGE
					{
					/* 000*/6,
					/* 001*/1,
					/* 010*/1,
					/* 011*/4,
					/* 100*/1,
					/* 101*/4,
					/* 110*/4,
					/* 111*/12 },

					// FACE
					{
					/* 000*/12,
					/* 001*/4,
					/* 010*/4,
					/* 011*/1,
					/* 100*/4,
					/* 101*/1,
					/* 110*/1,
					/* 111*/6 },

					// VOLUME
					{
					/* 000*/8,
					/* 001*/4,
					/* 010*/4,
					/* 011*/2,
					/* 100*/4,
					/* 101*/2,
					/* 110*/2,
					/* 111*/1 }

			};

	static constexpr size_t m_vertics_matrix_[4/* to iform*/][8/* node id*/][MAX_NUM_OF_CELL/*id shift*/] =
	{
	//To VERTEX
			{

			/* 000*/
			{ 0 },
			/* 001*/
			{ _LI, _HI },
			/* 010*/
			{ _LJ, _HJ },
			/* 011*/
			{ _LI | _LJ, _HI | _LJ, _HI | _HJ, _LI | _HJ },
			/* 100*/
			{ _LK, _HK },
			/* 101*/
			{ _LK | _LI, _HK | _LI, _HK | _HI, _LK | _HI },
			/* 110*/
			{ _LJ | _LK, _HJ | _LK, _HJ | _HK, _LJ | _HK },
			/* 111*/
			{ _LI | _LJ | _LK, //
			_HI | _LJ | _LK, //
			_HI | _HJ | _LK, //
			_LI | _HJ | _LK, //

			_LI | _LJ | _HK, //
			_HI | _LJ | _HK, //
			_HI | _HJ | _HK, //
			_LI | _HJ | _HK }

			},

			//To EDGE
			{
			/* 000*/
			{ _HI, _LI, _HJ, _LJ, _HK, _LK },
			/* 001*/
			{ 0 },
			/* 010*/
			{ 0 },
			/* 011*/
			{ _LJ, _HI, _HJ, _LI },
			/* 100*/
			{ 0 },
			/* 101*/
			{ _LI, _HK, _HI, _LK },
			/* 110*/
			{ _LK, _HJ, _HK, _LJ },
			/* 111*/
			{ _LK | _LJ,  //-> 001
			_LK | _HI, //   012
			_LK | _HJ, //   021
			_LK | _LI, //   010

			_LI | _LJ, //
			_LI | _HJ, //
			_HI | _LJ, //
			_HI | _HI, //

			_HK | _LJ, //
			_HK | _HI, //
			_HK | _HJ, //
			_HK | _LI  //
			} },

			//To FACE
			{
			/* 000*/
			{ _LK | _LJ,  //
			_LK | _HI, //
			_LK | _HJ, //
			_LK | _LI, //

			_LI | _LJ, //
			_LI | _HJ, //
			_HI | _LJ, //
			_HI | _HI, //

			_HK | _LJ, //
			_HK | _HI, //
			_HK | _HJ, //
			_HK | _LI  //
			},
			/* 001*/
			{ _LJ, _HK, _HJ, _LK },
			/* 010*/
			{ _LK, _HI, _HK, _LI },
			/* 011*/
			{ 0 },
			/* 100*/
			{ _LI, _HJ, _HI, _LJ },
			/* 101*/
			{ 0 },
			/* 110*/
			{ 0 },
			/* 111*/
			{ _LI, _LJ, _LK, _HI, _HJ, _HK } },
			// TO VOLUME
			{
			/* 000*/
			{ _LI | _LJ | _LK,  //
			_LI | _HJ | _LK, //
			_LI | _LJ | _HK, //
			_LI | _HJ | _HK, //

			_HI | _LJ | _LK, //
			_HI | _HJ | _LK, //
			_HI | _LJ | _HK, //
			_HI | _HJ | _HK //

			},
			/* 001*/
			{ _LJ | _LK, _LJ | _HK, _HJ | _LK, _HJ | _HK },
			/* 010*/
			{ _LK | _LI, _LK | _HI, _HK | _LI, _HK | _HI },
			/* 011*/
			{ _LK, _HK },
			/* 100*/
			{ _LI | _LJ, _LI | _HJ, _HI | _LJ, _HI | _HJ },
			/* 101*/
			{ _LJ, _HJ },
			/* 110*/
			{ _LI, _HI },
			/* 111*/
			{ 0 } }

	};
	template<size_t IFORM>
	static int get_adjoints(id_type s, id_type * res = nullptr)
	{
		int id = node_id(s);

		if (res != nullptr)
		{
			for (int i = 0; i < m_vertics_num_[IFORM][id]; ++i)
			{
				res[i] = ((s + m_vertics_matrix_[IFORM][id][i])
						& CLEAR_CARRAY_FLAG);
				;
			}
		}
		return m_vertics_num_[IFORM][id];
	}
	template<size_t IFORM>
	static int get_adjoints(id_type s, coordinates_type * res = nullptr)
	{
		int id = node_id(s);
		if (res != nullptr)
		{
			for (int i = 0; i < m_vertics_num_[IFORM][id]; ++i)
			{
				res[i] = id_to_coordinates(s + m_vertics_matrix_[IFORM][id][i]);
			}
		}
		return m_vertics_num_[IFORM][id];
	}

//	template<size_t IFORM> static constexpr int get_vertics(int n, id_type s,
//			coordinates_type *q = nullptr)
//	{
//		return get_vertics_(std::integral_constant<size_t, IFORM>(), n, s, q);
//	}
//	static int get_vertics_(std::integral_constant<size_t, VOLUME>, int n,
//			id_type s, coordinates_type *q = nullptr)
//	{
//
//		if (q != nullptr)
//		{
//			coordinates_type x0 = id_to_coordinates(s);
//
//			coordinates_type dx = id_to_coordinates(_DI | INDEX_ZERO);
//			coordinates_type dy = id_to_coordinates(_DJ | INDEX_ZERO);
//			coordinates_type dz = id_to_coordinates(_DK | INDEX_ZERO);
//
//			q[0] = x0 - dx - dy - dz;
//			q[1] = x0 + dx - dy - dz;
//			q[2] = x0 + dx + dy - dz;
//			q[3] = x0 - dx + dy - dz;
//
//			q[4] = x0 - dx - dy + dz;
//			q[5] = x0 + dx - dy + dz;
//			q[6] = x0 + dx + dy + dz;
//			q[7] = x0 - dx + dy + dz;
//		}
//
//		return 8;
//	}
//
//	static int get_vertics_(std::integral_constant<size_t, FACE>, int n,
//			id_type s, coordinates_type *q = nullptr)
//	{
//
//		if (q != nullptr)
//		{
//			coordinates_type x0 = id_to_coordinates(s);
//
//			coordinates_type d[3] = {
//
//			id_to_coordinates(_DI | INDEX_ZERO),
//
//			id_to_coordinates(_DJ | INDEX_ZERO),
//
//			id_to_coordinates(_DK | INDEX_ZERO) };
//
//			coordinates_type const &dx = d[(n + 1) % 3];
//			coordinates_type const &dy = d[(n + 2) % 3];
//			q[0] = x0 - dx - dy;
//			q[1] = x0 + dx - dy;
//			q[2] = x0 + dx + dy;
//			q[3] = x0 - dx + dy;
//		}
//
//		return 4;
//	}
//
//	static int get_vertics_(std::integral_constant<size_t, EDGE>, int n,
//			id_type s, coordinates_type *q = nullptr)
//	{
//
//		if (q != nullptr)
//		{
//			coordinates_type x0 = id_to_coordinates(s);
//
//			coordinates_type d[3] = {
//
//			id_to_coordinates(_DI | INDEX_ZERO),
//
//			id_to_coordinates(_DJ | INDEX_ZERO),
//
//			id_to_coordinates(_DK | INDEX_ZERO) };
//
//			coordinates_type const &dx = d[n];
//
//			q[0] = x0 - dx;
//			q[1] = x0 + dx;
//
//		}
//
//		return 4;
//	}

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
//	static constexpr id_type ele_suffix(id_type s)
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
//		return m_component_number_[ele_suffix(s)];
//	}
//
//
//	static constexpr id_type IForm(id_type r)
//	{
//		return m_iform_[ele_suffix(r)];
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
//		auto di = delta_index(rotate(dual(s)));
//		auto dj = delta_index(inverse_rotate(dual(s)));
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
//		auto d1 = delta_index(rotate(dual(s)));
//		auto d2 = delta_index(inverse_rotate(dual(s)));
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
//		auto d1 = delta_index(rotate((s)));
//		auto d2 = delta_index(inverse_rotate((s)));
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
//		auto d1 = delta_index(rotate((s)));
//		auto d2 = delta_index(inverse_rotate((s)));
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

/**
 * Solve problem: Undefined reference to static constexpr char[]
 * http://stackoverflow.com/questions/22172789/passing-a-static-constexpr-variable-by-universal-reference
 */

template<size_t N, size_t M, size_t A> constexpr size_t MeshIDs_<N, M, A>::FULL_DIGITS;
template<size_t N, size_t M, size_t A> constexpr size_t MeshIDs_<N, M, A>::ID_DIGITS;
template<size_t N, size_t M, size_t A> constexpr size_t MeshIDs_<N, M, A>::ID_MASK;
template<size_t N, size_t M, size_t A> constexpr size_t MeshIDs_<N, M, A>::_DK;
template<size_t N, size_t M, size_t A> constexpr size_t MeshIDs_<N, M, A>::_DJ;
template<size_t N, size_t M, size_t A> constexpr size_t MeshIDs_<N, M, A>::_DI;
template<size_t N, size_t M, size_t A> constexpr size_t MeshIDs_<N, M, A>::_DA;

template<size_t N, size_t M, size_t A> constexpr int MeshIDs_<N, M, A>::m_node_id_to_index_[];
template<size_t N, size_t M, size_t A> constexpr size_t MeshIDs_<N, M, A>::m_index_to_id_shift_[4][3];
template<size_t N, size_t M, size_t A> constexpr size_t MeshIDs_<N, M, A>::m_index_to_node_id_[4][3];
template<size_t N, size_t M, size_t A> constexpr typename MeshIDs_<N,M,A>::coordinates_type MeshIDs_<N,M,A>::m_index_to_coordinates_shift_[4][3];
template<size_t N, size_t M, size_t A> constexpr int MeshIDs_<N, M, A>::m_vertics_num_[4][8];
template<size_t N, size_t M, size_t A> constexpr size_t MeshIDs_<N, M, A>::m_vertics_matrix_[4/* to iform*/][8/* node id*/][MAX_NUM_OF_CELL/*id shift*/];

template<size_t N, size_t M, size_t A> constexpr typename MeshIDs_<N,M,A>::id_type MeshIDs_<N, M, A>::ID_ZERO;
template<size_t N, size_t M, size_t A> constexpr typename MeshIDs_<N,M,A>::index_type MeshIDs_<N, M, A>::INDEX_ZERO;

template<size_t N, size_t M, size_t A> constexpr Real MeshIDs_<N, M, A>::COORD_ZERO;

typedef MeshIDs_<3, 4> MeshIDs;

}
// namespace simpla

#endif /* CORE_MESH_MESH_IDS_H_ */

