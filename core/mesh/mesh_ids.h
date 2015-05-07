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

	typedef std::uint64_t id_type;

	typedef nTuple<id_type, 3> id_tuple;

	typedef long index_type;

	typedef nTuple<index_type, 3> index_tuple;

	typedef nTuple<Real, 3> coordinates_type;

	static constexpr int ndims = NDIMS;

	static constexpr id_type FULL_DIGITS = std::numeric_limits<id_type>::digits;

	static constexpr id_type ID_DIGITS = 20;

	static constexpr id_type HEAD_DIGITS = (FULL_DIGITS - ID_DIGITS * 3);

	static constexpr id_type ID_MASK = (1UL << ID_DIGITS) - 1;

	static constexpr id_type NO_HAED = (1UL << (ID_DIGITS * 3)) - 1;

	static constexpr id_type OVERFLOW_FLAG = (1UL) << (ID_DIGITS - 1);
	static constexpr id_type INDEX_ZERO = OVERFLOW_FLAG;
	static constexpr id_type ID_ZERO = INDEX_ZERO | (INDEX_ZERO << ID_DIGITS)
			| (INDEX_ZERO << (ID_DIGITS * 2));

	struct id_s
	{
		id_type i :20;
		id_type j :20;
		id_type k :20;
		int dummy :4;

		template<typename T>
		id_s &operator=(T const& x)
		{
			i = static_cast<id_type>(x[0]);
			j = static_cast<id_type>(x[1]);
			k = static_cast<id_type>(x[2]);
			return (*this);
		}

		template<typename T>
		id_s &operator=(nTuple<T, 4> const& x)
		{
			i = static_cast<id_type>(x[0]);
			j = static_cast<id_type>(x[1]);
			k = static_cast<id_type>(x[2]);

			*reinterpret_cast<id_type *>(this) |= m_id_to_shift_[x[3]];

			return (*this);
		}

		id_s &operator=(id_type s)
		{
			*reinterpret_cast<id_type *>(this) = s;

			return (*this);
		}
		operator id_type() const
		{
			return *reinterpret_cast<id_type const *>(this);
		}
		template<typename T>
		operator nTuple<T,3>() const
		{
			return nTuple<T, 3>( {

			static_cast<T>(i),

			static_cast<T>(j),

			static_cast<T>(k)

			});
		}
		template<typename T>
		operator nTuple<T,4>() const
		{
			return nTuple<T, 4>( {

			static_cast<T>(i),

			static_cast<T>(j),

			static_cast<T>(k),

			m_id_to_index_[node_id(raw_cast<id_type const>(*this))],

			});
		}
	};
	/// @}
	/// @name level dependent
	/// @{

	static constexpr int MESH_LEVEL = IMESH_LEVEL;

	static constexpr id_type SUB_ID_MASK = ((1UL << MESH_LEVEL) - 1);

	static constexpr id_type _D = 1UL << (MESH_LEVEL - 1);

	static constexpr Real _R = static_cast<Real>(_D);

//#ifdef BIG_ENDIAN

	static constexpr id_type _DI = _D;

	static constexpr id_type _DJ = _D << (ID_DIGITS);

	static constexpr id_type _DK = _D << (ID_DIGITS * 2);

	static constexpr id_type PRIMARY_ID_MASK_ = ID_MASK & (~SUB_ID_MASK);

	static constexpr id_type PRIMARY_ID_MASK = PRIMARY_ID_MASK_
			| (PRIMARY_ID_MASK_ << ID_DIGITS)
			| (PRIMARY_ID_MASK_ << (ID_DIGITS * 2));

//#else
//	static constexpr id_type _DK = _D << (HEAD_DIGITS);
//
//	static constexpr id_type _DJ = _D << (ID_DIGITS + HEAD_DIGITS);
//
//	static constexpr id_type _DI = _D << (ID_DIGITS * 2 + HEAD_DIGITS);
//
//	static constexpr id_type ID_ZERO = (OVERFLOW_FLAG<<HEAD_DIGITS)
//	| (OVERFLOW_FLAG << (ID_DIGITS+HEAD_DIGITS)) | (OVERFLOW_FLAG << (ID_DIGITS * 2+HEAD_DIGITS));
//
//	static constexpr id_type PRIMARY_ID_MASK = (PRIMARY_ID_MASK_
//			| (PRIMARY_ID_MASK_ << ID_DIGITS)
//			| (PRIMARY_ID_MASK_ << (ID_DIGITS * 2)))<< (HEAD_DIGITS);
//
//#endif
	static constexpr id_type _DA = _DI | _DJ | _DK;

	static constexpr Real COORDINATES_MESH_FACTOR = static_cast<Real>(1UL
			<< MESH_LEVEL);

	/// @}

	static constexpr id_type m_sub_index_to_id_[4][3] = { //

			{ 0, 0, 0 }, /*VERTEX*/
			{ 1, 2, 4 }, /*EDGE*/
			{ 6, 5, 3 }, /*FACE*/
			{ 7, 7, 7 } /*VOLUME*/

			};

	static constexpr id_type m_id_to_sub_index_[8] = { //

			0, // 000
					0, // 001
					1, // 010
					2, // 011
					2, // 100
					1, // 101
					0, // 110
					0, // 111
			};

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

	static constexpr id_type m_id_to_iform_[] = { //

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
	static constexpr id_type sub_id(int n = 0)
	{
		return m_sub_index_to_id_[IFORM][n];
	}

	template<typename T>
	static constexpr id_type pack(T const & idx)
	{
		return static_cast<id_type>(assign_cast<id_s>(idx));
	}

	static constexpr id_tuple unpack(id_type s)
	{
		return static_cast<id_tuple>(raw_cast<id_s>(s));
	}

	template<typename T>
	static constexpr id_type pack_index(T const & idx, int n_id = 0)
	{
		return (pack(idx) << MESH_LEVEL) | m_id_to_shift_[n_id];
	}

	static constexpr id_tuple unpack_index(id_type s)
	{
		return unpack(s) >> MESH_LEVEL;
	}
	template<typename T>
	static constexpr T type_cast(id_type s)
	{
		return static_cast<T>(raw_cast<id_s const>(s));
	}

	static constexpr id_type const &id(id_type const &s)
	{
		return s;
	}

	static constexpr coordinates_type coordinates(id_type s)
	{
		return static_cast<coordinates_type>(raw_cast<id_s>(s));
	}

	static constexpr index_tuple index(id_type const &s)
	{
		return (unpack(s) >> MESH_LEVEL);
	}

	static constexpr int num_of_ele_in_cell(id_type s)
	{
		return m_id_to_num_of_ele_in_cell_[node_id(s)];
	}

	template<typename TX>
	static std::tuple<id_type, coordinates_type> coordinates_global_to_local(
			TX const &x, int n_id = 0)
	{

		id_type s = (pack(x - m_id_to_coordinates_shift_[n_id])
				& PRIMARY_ID_MASK) | m_id_to_shift_[n_id];

		coordinates_type r;

		r = (x - coordinates(s)) / (_R * 2.0);

		return std::make_tuple(s, r);

	}

	static constexpr coordinates_type coordinates_local_to_global(
			std::tuple<id_type, coordinates_type> const &t)
	{
		return coordinates(std::get<0>(t)) + std::get<1>(t);
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
		return ((s & (~_DA))
				| (((s & (_DA)) << ID_DIGITS) | ((s & _DK) >> (ID_DIGITS * 2))))
				& NO_HAED;
	}

	static constexpr id_type inverse_rotate(id_type const &s)
	{
		return ((s & (~_DA))
				| (((s & (_DA)) >> ID_DIGITS) | ((s & _DI) << (ID_DIGITS * 2))))
				& NO_HAED;
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

	static constexpr id_type m_id_to_index_[8] = { //

			0, // 000
					0, // 001
					1, // 010
					2, // 011
					2, // 100
					1, // 101
					0, // 110
					0, // 111
			};

	static constexpr id_type sub_index(id_type const &s)
	{
		return m_id_to_index_[node_id(s)];
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
	static constexpr id_type _LI = (((-_D) & ID_MASK));
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

		typedef size_t difference_type;

		struct iterator;

		typedef iterator const_iterator;

		typedef range_type this_type;

		template<typename T0, typename T1>
		range_type(T0 const & min, T1 const & max)
				: m_min_(pack_index(min)), m_max_(pack_index(max))
		{

		}

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
			return ++const_iterator(m_min_, m_max_,
					inverse_rotate(m_max_ - (_DA << 1)));
		}

		const_iterator rbegin() const
		{
			return const_iterator(m_min_, m_max_,
					inverse_rotate(m_max_ - (_DA << 1)));
		}

		const_iterator rend() const
		{
			const_iterator res(m_min_, m_max_,
					inverse_rotate(m_min_ - (_DA << 1)));

			++res;
			return std::move(res);
		}

		std::tuple<nTuple<index_type, ndims + 1>, nTuple<index_type, ndims + 1>> box() const
		{
			nTuple<index_type, ndims + 1> b, e;
			b = unpack_index(m_min_);
			e = unpack_index(m_max_);
			b[ndims] = 0;
			e[ndims] = m_id_to_num_of_ele_in_cell_[node_id(m_min_)];
			return std::make_tuple(b, e);
		}

		template<typename T>
		bool in_box(T const & s) const
		{
			return in_box(pack(s));
		}
		bool in_box(id_type s) const
		{
			UNIMPLEMENTED;

			return true;
		}
		bool empty() const
		{
			return m_min_ == m_max_;
		}
		void clear()
		{
			m_min_ = m_max_;
		}

		difference_type size() const
		{
			return NProduct(index(m_max_ - m_min_))
					* m_id_to_num_of_ele_in_cell_[node_id(m_min_)];
		}

		template<typename ...Args>
		void reset(Args && ...args)
		{
			this_type(pack(args)...).swap(*this);
		}

		struct iterator: public std::iterator<
				typename std::bidirectional_iterator_tag, id_type,
				difference_type>
		{
		private:
			id_type m_min_, m_max_, m_self_;
		public:
			iterator(id_type const & min, id_type const & max,
					id_type const& self)
					: m_min_(min), m_max_(max), m_self_(self)
			{

			}
			iterator(id_type const & min, id_type const & max)
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
		private:

#define carray(  flag,  min,  max,   self)          \
			{                                                                \
				std::div_t  div = std::div(self + flag*(_D<<1) + max - min * 2, max - min);      \
				self = div.rem + min;      \
				flag= div.quot - 1;                                         \
		}

		public:
			void next()
			{
				m_self_ = rotate(m_self_);
				if (sub_index(m_self_) == 0)
				{
					id_s const & min = raw_cast<id_s>(m_min_);
					id_s const & max = raw_cast<id_s>(m_max_);
					id_s & self = raw_cast<id_s>(m_self_);
					int flag = 1;

					carray(flag, min.k, max.k, self.k);
					carray(flag, min.j, max.j, self.j);
					self.i += flag * (_D << 1);
//					carray(flag, min.i, max.i, self.i);
				}

			}
			void prev()
			{
				m_self_ = inverse_rotate(m_self_);
				if (sub_index(m_self_) == 0)
				{
					id_s const & min = raw_cast<id_s>(m_min_);
					id_s const & max = raw_cast<id_s>(m_max_);
					id_s & self = raw_cast<id_s>(m_self_);

					int flag = -1;
					carray(flag, min.k, max.k, &self.k);
					carray(flag, min.j, max.j, &self.j);
//					carray(flag, min.i, max.i, &self.i);
					self.i += flag * (_D << 1);

				}
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
	static constexpr range_type make_range(id_type b, id_type e)
	{
		return range_type(b | m_id_to_shift_[m_sub_index_to_id_[IFORM][0]],
				e | m_id_to_shift_[m_sub_index_to_id_[IFORM][0]]);
	}

	template<size_t IFORM, typename T0, typename T1>
	static constexpr range_type make_range(T0 const & b, T1 const & e)
	{
		return range_type(id(b) | m_id_to_shift_[m_sub_index_to_id_[IFORM][0]],
				id(e) | m_id_to_shift_[m_sub_index_to_id_[IFORM][0]]);
	}

	static constexpr size_t hash(id_type s, id_type b, id_type e)
	{

		return hash_(raw_cast<id_s>(s - b), raw_cast<id_s>(e - b))
				* num_of_ele_in_cell(s) + sub_index(s);
	}

	static constexpr size_t hash_(id_s const & s, id_s const & d)
	{
		//C-ORDER SLOW FIRST

		return (s.k >> MESH_LEVEL)
				+ ((s.j >> MESH_LEVEL)
						+ (s.i >> MESH_LEVEL) * (d.j >> MESH_LEVEL))
						* (d.k >> MESH_LEVEL);;
	}
	template<size_t IFORM>
	static constexpr size_t max_hash(id_type b, id_type e)
	{
		return NProduct(unpack_index(e - b))
				* m_id_to_num_of_ele_in_cell_[sub_id<IFORM>(0)];
	}
}
;

/**
 * Solve problem: Undefined reference to static constexpr char[]
 * http://stackoverflow.com/questions/22172789/passing-a-static-constexpr-variable-by-universal-reference
 */

template<size_t N, size_t M> constexpr int MeshIDs_<N, M>::ndims;
template<size_t N, size_t M> constexpr int MeshIDs_<N, M>::MESH_LEVEL;

template<size_t N, size_t M> constexpr typename MeshIDs_<N, M >::id_type MeshIDs_<N, M >::OVERFLOW_FLAG;
template<size_t N, size_t M> constexpr typename MeshIDs_<N, M >::id_type MeshIDs_<N, M >::ID_ZERO;
template<size_t N, size_t M> constexpr typename MeshIDs_<N, M >::id_type MeshIDs_<N, M >::INDEX_ZERO;

template<size_t N, size_t M> constexpr typename MeshIDs_<N, M >::id_type MeshIDs_<N, M >::FULL_DIGITS;
template<size_t N, size_t M> constexpr typename MeshIDs_<N, M >::id_type MeshIDs_<N, M >::ID_DIGITS;
template<size_t N, size_t M> constexpr typename MeshIDs_<N, M >::id_type MeshIDs_<N, M >::ID_MASK;
template<size_t N, size_t M> constexpr typename MeshIDs_<N, M >::id_type MeshIDs_<N, M >::_DK;
template<size_t N, size_t M> constexpr typename MeshIDs_<N, M >::id_type MeshIDs_<N, M >::_DJ;
template<size_t N, size_t M> constexpr typename MeshIDs_<N, M >::id_type MeshIDs_<N, M >::_DI;
template<size_t N, size_t M> constexpr typename MeshIDs_<N, M >::id_type MeshIDs_<N, M >::_DA;
template<size_t N, size_t M> constexpr typename MeshIDs_<N, M >::id_type MeshIDs_<N, M >::m_id_to_index_[];
template<size_t N, size_t M> constexpr typename MeshIDs_<N, M >::id_type MeshIDs_<N, M >::m_id_to_shift_[];
template<size_t N, size_t M> constexpr typename MeshIDs_<N, M >::id_type MeshIDs_<N, M >::m_sub_index_to_id_[4][3];
template<size_t N, size_t M> constexpr typename MeshIDs_<N, M >::id_type MeshIDs_<N, M >::m_id_to_num_of_ele_in_cell_[];
template<size_t N, size_t M> constexpr typename MeshIDs_<N,M >::coordinates_type MeshIDs_<N,M >::m_id_to_coordinates_shift_[ ];
template<size_t N, size_t M> constexpr int MeshIDs_<N, M>::m_vertics_num_[4][8];
template<size_t N, size_t M> constexpr typename MeshIDs_<N, M >::id_type MeshIDs_<N, M>::m_vertics_matrix_[4/* to iform*/][8/* node id*/][MAX_NUM_OF_CELL/*id shift*/];

typedef MeshIDs_<3, 4> MeshIDs;

}
// namespace simpla

#endif /* CORE_MESH_MESH_IDS_H_ */

