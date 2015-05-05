/**
 * @file mesh_ids.h
 *
 * @date 2015年3月19日
 * @author salmon
 */

#ifndef CORE_MESH_MESH_IDS_H_
#define CORE_MESH_MESH_IDS_H_

#include <stddef.h>
#include <limits>
#include <tuple>

#include "../gtl/ntuple.h"
#include "../gtl/primitives.h"

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
template<size_t NDIMS = 3, size_t IMESH_LEVEL = 4>
struct MeshIDs_
{
	/// @name level independent
	/// @{

	typedef std::int16_t short_id_type;

	typedef nTuple<short_id_type, 4> id_tuple;

	typedef std::uint64_t id_type;

	typedef std::int64_t index_type;

	typedef nTuple<index_type, 3> index_tuple;

	typedef nTuple<Real, 3> coordinates_type;

	static constexpr size_t ndims = NDIMS;

	static constexpr size_t ID_DIGITS =
			std::numeric_limits<std::uint16_t>::digits;

	static constexpr id_type ID_MASK = (1UL << (ID_DIGITS)) - 1;
	static constexpr short_id_type OVERFLOW_FLAG = (1UL << (ID_DIGITS - 1));
	/// @}
	/// @name level dependent
	/// @{

	static constexpr size_t MESH_LEVEL = IMESH_LEVEL;

	static constexpr id_type SUB_ID_MASK = ((1UL << MESH_LEVEL) - 1);

	static constexpr id_type PRIMARY_ID_MASK = ID_MASK & (~SUB_ID_MASK);

	static constexpr id_type _D = (1UL << (MESH_LEVEL - 1));

	static constexpr Real _R = static_cast<Real>(_D);

#ifdef BIG_ENDIAN

	static constexpr id_type _DI = _D;	//| (((1UL << 63) - 1) << 16);

	static constexpr id_type _DJ = (_D << (ID_DIGITS));	//	| (((1UL << 63) - 1) << 32);

	static constexpr id_type _DK = _D << (ID_DIGITS * 2);//	| (((1UL << 63) - 1) << 48);

#else
	static constexpr id_type _DI = _D << (ID_DIGITS * 2);

	static constexpr id_type _DJ = _D << (ID_DIGITS);

	static constexpr id_type _DK = _D;

#endif

	static constexpr id_type _DA = _DI | _DJ | _DK;

	static constexpr Real COORDINATES_MESH_FACTOR = static_cast<Real>(1UL
			<< MESH_LEVEL);

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

	static constexpr size_t m_sub_index_to_id_[4][3] = { //

			{ 0, 0, 0 }, /*VERTEX*/
			{ 1, 2, 4 }, /*EDGE*/
			{ 6, 5, 3 }, /*FACE*/
			{ 7, 7, 7 } /*VOLUME*/

			};
	static constexpr size_t m_id_to_sub_index_[8] = { //

			0, // 000
					0, // 001
					1, // 010
					2, // 011
					2, // 100
					1, // 101
					0, // 110
					0, // 111
			};
//	static constexpr id_type m_id_to_iform_[] = { //
//
//			VERTEX, // 000
//					EDGE, // 001
//					EDGE, // 010
//					FACE, // 011
//					EDGE, // 100
//					FACE, // 101
//					FACE, // 110
//					VOLUME // 111
//			};
	static constexpr id_type m_id_to_shift_[] = {

	0,   					// 000
			_DI,   			// 001
			_DJ, 			// 010
			(_DI | _DJ), 	// 011
			_DK, 			// 100
			(_DK | _DI), 	// 101
			(_DJ | _DK), 	// 110
			_DA 			// 111

			};

	static constexpr coordinates_type m_id_to_coordinates_shift_[] = {

	{ 0, 0, 0 },            // 000
			{ _R, 0, 0 },           // 001
			{ 0, _R, 0 },           // 010
			{ 0, 0, _R },           // 011
			{ _R, _R, 0 },          // 100
			{ _R, 0, _R },          // 101
			{ 0, _R, _R },          // 110
			{ 0, _R, _R },          // 111

			};

	static constexpr id_type m_id_to_num_of_ele_in_cell_[] = {

	1,   		// 000
			3,	// 001
			3,	// 010
			3,	// 011
			3,	// 100
			3, 	// 101
			3, 	// 110
			1 	// 111
			};

	template<typename T>
	static constexpr id_type pack(T const & idx)
	{
		return static_cast<id_type>(id_tuple(
				{ static_cast<short_id_type>(idx[0] + OVERFLOW_FLAG),
						static_cast<short_id_type>(idx[1] + OVERFLOW_FLAG),
						static_cast<short_id_type>(idx[2] + OVERFLOW_FLAG) }));
	}

	static constexpr id_tuple unpack(id_type const &s)
	{
		return raw_cast<id_tuple const>(s) - OVERFLOW_FLAG;
	}

	static constexpr id_type id(coordinates_type const &x)
	{
		return pack(x);
	}

	static constexpr coordinates_type coordinates(id_type s)
	{
		return static_cast<coordinates_type>(unpack(s));
	}

	template<typename T>
	static constexpr id_type pack_diff(T const & idx)
	{
		return pack(idx) << MESH_LEVEL;

	}

	static constexpr index_tuple unpack_diff(id_type s)
	{
		return unpack(s) >> MESH_LEVEL;
	}

	template<size_t IFORM, typename TX>
	static std::tuple<id_type, coordinates_type> coordinates_global_to_local(
			TX const &x, int n = 0)
	{
		id_type id = code(
				x - m_id_to_coordinates_shift_[m_sub_index_to_id_[IFORM][n]]);

		coordinates_type r;

		r = (x - unpack(id)) / (_R * 2.0);

		return std::make_tuple(id, r);

	}

	static constexpr size_t hash(id_type offset, index_tuple const & dims,
			index_tuple const & strides)
	{
		return inner_product((unpack_diff(offset) + dims) % dims, strides)
				* m_id_to_num_of_ele_in_cell_[node_id(offset)]
				+ m_id_to_sub_index_[node_id(offset)];
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

	static constexpr size_t sub_index(id_type const &s)
	{
		return m_id_to_sub_index_[node_id(s)];
	}
	static constexpr size_t iform(id_type const &s)
	{
		return ((s >> (MESH_LEVEL - 1)) & 1UL)
				+ ((s >> (ID_DIGITS + MESH_LEVEL - 1)) & 1UL)
				+ ((s >> (ID_DIGITS * 2 + MESH_LEVEL - 1)) & 1UL);
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

	static constexpr id_type _HI = _D;
	static constexpr id_type _HJ = _HI << ID_DIGITS;
	static constexpr id_type _HK = _HI << (ID_DIGITS * 2);

	static constexpr id_type _LI = ((-_D) & ID_MASK);
	static constexpr id_type _LJ = _LI << ID_DIGITS;
	static constexpr id_type _LK = _LI << (ID_DIGITS * 2);

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

	static constexpr id_type m_vertics_matrix_[4/* to iform*/][8/* node id*/][MAX_NUM_OF_CELL/*id shift*/] =
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
				res[i] = ((s + m_vertics_matrix_[IFORM][id][i]));
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
				res[i] = coordinates(s + m_vertics_matrix_[IFORM][id][i]);
			}
		}
		return m_vertics_num_[IFORM][id];
	}

	struct range_type
	{

	private:
		id_type m_min_, m_max_;
	public:
		typedef id_type value_type;

		typedef index_type difference_type;

		struct iterator;

		typedef iterator const_iterator;

		typedef range_type this_type;

		range_type(id_type const & min, id_type const & max)
				: m_min_(min), m_max_(max)
		{

		}

		range_type(this_type const & other)
				: m_min_(other.m_min_), m_max_(other.m_max_)
		{

		}
		~range_type()
		{

		}
		this_type &operator=(this_type const & other)
		{
			this_type(other).swap(*this);
			return *this;
		}
		this_type operator&(this_type const & other) const
		{
			return *this;
		}
		void swap(this_type & other)
		{
			std::swap(m_min_, other.m_min_);
			std::swap(m_max_, other.m_max_);
		}

		const_iterator begin() const
		{
			return const_iterator(m_min_, m_max_, m_min_);
		}

		const_iterator end() const
		{
			return const_iterator(m_min_, m_max_, m_max_);
		}
		std::tuple<id_type, id_type> box() const
		{
			return std::make_tuple(m_min_, m_max_);
		}
		bool in_box(id_type s) const
		{
			UNIMPLEMENTED;

			return true;
		}
		bool is_empty() const
		{
			return m_min_ == m_max_;
		}
		void clear()
		{
			m_min_ = m_max_;
		}
		difference_type size() const
		{
			return NProduct(unpack_diff(m_max_ - m_min_))
					* m_id_to_num_of_ele_in_cell_[node_id(m_min_)];
		}

		struct iterator: public std::iterator<
				typename std::bidirectional_iterator_tag, value_type,
				difference_type>
		{
		private:
			value_type m_min_, m_max_, m_self_;
		public:
			iterator(value_type const & min, value_type const & max,
					value_type const& self)
					: m_min_(min), m_max_(max), m_self_(self)
			{

			}
			iterator(value_type const & min, value_type const & max)
					: m_min_(min), m_max_(max), m_self_(min)
			{

			}
			~iterator()
			{

			}
			typedef iterator this_type;
			bool operator==(this_type const & other) const
			{
				return m_self_ == other.m_self_ && (m_min_ == other.m_min_)
						&& (m_max_ == other.m_max_);
			}
			bool operator!=(this_type const & other) const
			{
				return m_self_ != other.m_self_ || (m_min_ != other.m_min_)
						|| (m_max_ != other.m_max_);
			}

			value_type const & operator *() const
			{
				return m_self_;
			}
			void next()
			{
				UNIMPLEMENTED;
			}
			void prev()
			{
				UNIMPLEMENTED;
			}
			this_type & operator++()
			{
				next();
				return *this;
			}

			this_type & operator--()
			{
				prev();
				return *this;
			}

			this_type operator++(int)
			{
				this_type res(*this);
				++(*this);
				return std::move(res);
			}
			this_type operator--(int)
			{
				this_type res(*this);
				--(*this);
				return std::move(res);
			}

		};

	};
	template<size_t IFORM>
	static constexpr range_type range(id_type b, id_type e)
	{
		return range_type(
				(b & PRIMARY_ID_MASK)
						| m_id_to_shift_[m_sub_index_to_id_[IFORM][0]],

				(e & PRIMARY_ID_MASK)
						| m_id_to_shift_[m_sub_index_to_id_[IFORM][0]]

						);
	}

	template<size_t IFORM>
	static constexpr range_type range(std::tuple<id_type, id_type> const & box)
	{
		return range_type(std::get<0>(box), std::get<1>(box));
	}

}
;

/**
 * Solve problem: Undefined reference to static constexpr char[]
 * http://stackoverflow.com/questions/22172789/passing-a-static-constexpr-variable-by-universal-reference
 */
template<size_t N, size_t M> constexpr size_t MeshIDs_<N, M>::ndims;
template<size_t N, size_t M> constexpr size_t MeshIDs_<N, M>::MESH_LEVEL;

template<size_t N, size_t M> constexpr size_t MeshIDs_<N, M>::ID_DIGITS;
template<size_t N, size_t M> constexpr size_t MeshIDs_<N, M>::ID_MASK;

template<size_t N, size_t M> constexpr typename MeshIDs_<N, M>::id_type MeshIDs_<N, M>::_DK;
template<size_t N, size_t M> constexpr typename MeshIDs_<N, M>::id_type MeshIDs_<N, M>::_DJ;
template<size_t N, size_t M> constexpr typename MeshIDs_<N, M>::id_type MeshIDs_<N, M>::_DI;
template<size_t N, size_t M> constexpr typename MeshIDs_<N, M>::id_type MeshIDs_<N, M>::_DA;
template<size_t N, size_t M> constexpr typename MeshIDs_<N, M>::short_id_type MeshIDs_<N, M>::OVERFLOW_FLAG;

template<size_t N, size_t M> constexpr typename MeshIDs_<N, M>::id_type MeshIDs_<N, M>::m_sub_index_to_id_[4][3];
template<size_t N, size_t M> constexpr size_t MeshIDs_<N, M>::m_id_to_sub_index_[];
template<size_t N, size_t M> constexpr size_t MeshIDs_<N, M>::m_id_to_num_of_ele_in_cell_[];
template<size_t N, size_t M> constexpr size_t MeshIDs_<N, M>::m_id_to_shift_[];
template<size_t N, size_t M> constexpr typename MeshIDs_<N,M >::coordinates_type MeshIDs_<N,M >::m_id_to_coordinates_shift_[];

template<size_t N, size_t M> constexpr int MeshIDs_<N, M>::m_vertics_num_[4][8];
template<size_t N, size_t M> constexpr size_t MeshIDs_<N, M>::m_vertics_matrix_[4/* to iform*/][8/* node id*/][MAX_NUM_OF_CELL/*id shift*/];

typedef MeshIDs_<3, 4> MeshIDs;

}
// namespace simpla

#endif /* CORE_MESH_MESH_IDS_H_ */

