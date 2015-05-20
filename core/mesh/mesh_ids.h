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

	static constexpr id_type ID_DIGITS = 21;

	static constexpr id_type HEAD_DIGITS = (FULL_DIGITS - ID_DIGITS * 3);

	static constexpr id_type ID_MASK = (1UL << ID_DIGITS) - 1;

	static constexpr id_type NO_HAED = (1UL << (ID_DIGITS * 3)) - 1;

	static constexpr id_type OVERFLOW_FLAG = (1UL) << (ID_DIGITS - 1);

	static constexpr id_type FULL_OVERFLOW_FLAG = OVERFLOW_FLAG
			| (OVERFLOW_FLAG << ID_DIGITS) | (OVERFLOW_FLAG << (ID_DIGITS * 2));

	static constexpr id_type INDEX_ZERO = (1UL) << (ID_DIGITS - 2);

	static constexpr id_type ID_ZERO = INDEX_ZERO | (INDEX_ZERO << ID_DIGITS)
			| (INDEX_ZERO << (ID_DIGITS * 2));

	static constexpr Real EPSILON = 1.0 / static_cast<Real>(INDEX_ZERO);

	/// @}

	/// @name level dependent
	/// @{

	static constexpr int MESH_LEVEL = IMESH_LEVEL;

	static constexpr id_type SUB_ID_MASK = ((1UL << MESH_LEVEL) - 1);

	static constexpr id_type FULL_SUB_ID_MASK = SUB_ID_MASK
			| (SUB_ID_MASK << ID_DIGITS) | (SUB_ID_MASK << (ID_DIGITS * 2));

	static constexpr id_type _D = 1UL << (MESH_LEVEL - 1);

	static constexpr Real _R = static_cast<Real>(_D);

	static constexpr id_type _DI = _D;

	static constexpr id_type _DJ = _D << (ID_DIGITS);

	static constexpr id_type _DK = _D << (ID_DIGITS * 2);

	static constexpr id_type PRIMARY_ID_MASK_ = ID_MASK & (~SUB_ID_MASK);

	static constexpr id_type PRIMARY_ID_MASK = PRIMARY_ID_MASK_
			| (PRIMARY_ID_MASK_ << ID_DIGITS)
			| (PRIMARY_ID_MASK_ << (ID_DIGITS * 2));

	static constexpr id_type _DA = _DI | _DJ | _DK;

	static constexpr Real COORDINATES_MESH_FACTOR = static_cast<Real>(1UL
			<< MESH_LEVEL);

	/// @}
	static constexpr Vec3 dx()
	{
		return Vec3( { COORDINATES_MESH_FACTOR, COORDINATES_MESH_FACTOR,
				COORDINATES_MESH_FACTOR });
	}

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
			_DI,   					// 001
			_DJ,   					// 010
			(_DI | _DJ),   					// 011
			_DK,   					// 100
			(_DK | _DI),   					// 101
			(_DJ | _DK),   					// 110
			_DA   					// 111

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
			3,   		// 001
			3,   		// 010
			3,   		// 011
			3,   		// 100
			3,   		// 101
			3,   		// 110
			1   		// 111
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
	static constexpr id_type sub_index_to_id(int n = 0)
	{
		return m_sub_index_to_id_[IFORM][n];
	}

	static constexpr int iform(id_type s)
	{
		return m_id_to_iform_[node_id(s)];
	}

	static constexpr id_type diff(id_type a, id_type b)
	{
		return ((a | OVERFLOW_FLAG) - b) & (~OVERFLOW_FLAG);
	}
	static constexpr id_type diff_index(id_type a, id_type b)
	{
		return diff(a, b) & (PRIMARY_ID_MASK);
	}

#define UNPACK_ID(_S_,_I_)    (static_cast<id_type>(_S_) >> (ID_DIGITS*(_I_ )) & ID_MASK)
#define UNPACK_INDEX(_S_,_I_)   static_cast<index_type>((_S_>> (ID_DIGITS*(_I_ )) & ID_MASK)>>MESH_LEVEL)

	template<typename T>
	static constexpr id_type pack(T const & idx)
	{
		return (static_cast<id_type>(idx[0]) & ID_MASK)
				| ((static_cast<id_type>(idx[1]) & ID_MASK) << ID_DIGITS)
				| ((static_cast<id_type>(idx[2]) & ID_MASK) << (ID_DIGITS * 2));
	}

	static constexpr id_tuple unpack(id_type s)
	{
		return id_tuple( {

		UNPACK_ID(s, 0),

		UNPACK_ID(s, 1),

		UNPACK_ID(s, 2)

		});;
	}

	static constexpr index_tuple unpack_index(id_type s)
	{
		return index_tuple( {

		UNPACK_INDEX(s, 0),

		UNPACK_INDEX(s, 1),

		UNPACK_INDEX(s, 2)

		});

	}

	template<typename T>
	static constexpr id_type pack_index(T const & idx, int n_id = 0)
	{
		return (pack(idx) << MESH_LEVEL) | m_id_to_shift_[n_id];
	}

	template<typename T>
	static constexpr T type_cast(id_type s)
	{
		return static_cast<T>(unpack(s));
	}

	static constexpr coordinates_type coordinates(id_type s)
	{
		return static_cast<coordinates_type>(unpack(s));
	}

	static constexpr int num_of_ele_in_cell(id_type s)
	{
		return m_id_to_num_of_ele_in_cell_[node_id(s)];
	}

	template<typename TX>
	static std::tuple<id_type, coordinates_type> coordinates_global_to_local(
			TX const &x, int n_id = 0)
	{

		id_type s = (pack(x - m_id_to_coordinates_shift_[n_id] + _R)
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
					_LK | _HI,  //   012
					_LK | _HJ,  //   021
					_LK | _LI,  //   010

					_LI | _LJ,  //
					_LI | _HJ,  //
					_HI | _LJ,  //
					_HI | _HI,  //

					_HK | _LJ,  //
					_HK | _HI,  //
					_HK | _HJ,  //
					_HK | _LI  //
					} },

					//To FACE
					{
					/* 000*/
					{ _LK | _LJ,  //
					_LK | _HI,  //
					_LK | _HJ,  //
					_LK | _LI,  //

					_LI | _LJ,  //
					_LI | _HJ,  //
					_HI | _LJ,  //
					_HI | _HI,  //

					_HK | _LJ,  //
					_HK | _HI,  //
					_HK | _HJ,  //
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
					_LI | _HJ | _LK,  //
					_LI | _LJ | _HK,  //
					_LI | _HJ | _HK,  //

					_HI | _LJ | _LK,  //
					_HI | _HJ | _LK,  //
					_HI | _LJ | _HK,  //
					_HI | _HJ | _HK  //

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

//	private:

		id_type m_min_, m_max_;
	public:

		typedef id_type value_type;

		typedef size_t difference_type;

		struct iterator;

		typedef iterator const_iterator;

		typedef range_type this_type;

		template<typename T0, typename T1>
		range_type(T0 const & min, T1 const & max, int n_id = 0)
				: m_min_(pack_index(min) | m_id_to_shift_[n_id]), m_max_(
						pack_index(max) | m_id_to_shift_[n_id])
		{

		}

		range_type(id_type const & min, id_type const & max, int n_id = 0)
				: m_min_(min | m_id_to_shift_[n_id]), m_max_(
						max | m_id_to_shift_[n_id])
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

//		const_iterator rbegin() const
//		{
//			return const_iterator(m_min_, m_max_,
//					inverse_rotate(m_max_ - (_DA << 1)));
//		}
//
//		const_iterator rend() const
//		{
//			const_iterator res(m_min_, m_max_,
//					inverse_rotate(m_min_ - (_DA << 1)));
//
//			++res;
//			return std::move(res);
//		}

		auto box() const
		DECL_RET_TYPE(std::forward_as_tuple(m_min_, m_max_))

		template<typename T>
		bool in_box(T const & x) const
		{
			auto b = unpack_index(m_min_);
			auto e = unpack_index(m_max_);
			return (b[1] <= x[1]) && (b[2] <= x[2]) && (b[0] <= x[0])  //
					&& (e[1] > x[1]) && (e[2] > x[2]) && (e[0] > x[0]);

		}
		constexpr bool in_box(id_type s) const
		{
			return in_box(unpack_index(s));
		}
		constexpr bool empty() const
		{
			return m_min_ == m_max_;
		}
		void clear()
		{
			m_min_ = m_max_;
		}

		constexpr difference_type size() const
		{
			return NProduct(unpack_index(m_max_ - m_min_))
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

			iterator(iterator const & other)
					: m_min_(other.m_min_), m_max_(other.m_max_), m_self_(
							other.m_self_)
			{
			}

			~iterator()
			{

			}

			typedef iterator this_type;

			bool operator==(this_type const & other) const
			{
				return m_self_ == other.m_self_;
			}
			bool operator!=(this_type const & other) const
			{
				return m_self_ != other.m_self_;
			}

			value_type const & operator *() const
			{
				return m_self_;
			}
		private:

			index_type carray_(id_type * self, id_type min, id_type max,
					index_type flag = 0)
			{

				auto div = std::div(
						static_cast<long>(*self + flag * (_D << 1) + max
								- min * 2), static_cast<long>(max - min));

				*self = static_cast<id_type>(div.rem + min);

				return div.quot - 1L;
			}

			index_type carray(id_type * self, id_type xmin, id_type xmax,
					index_type flag = 0)
			{
				id_tuple idx, min, max;

				idx = unpack(*self);
				min = unpack(xmin);
				max = unpack(xmax);

				flag = carray_(&idx[0], min[0], max[0], flag);
				flag = carray_(&idx[1], min[1], max[1], flag);
				flag = carray_(&idx[2], min[2], max[2], flag);

				*self = pack(idx) | (std::abs(flag) << (FULL_DIGITS - 1));
				return flag;
			}

		public:
			void next()
			{
				m_self_ = rotate(m_self_);
				if (sub_index(m_self_) == 0)
				{
					carray(&m_self_, m_min_, m_max_, 1);
				}

			}
			void prev()
			{
				m_self_ = inverse_rotate(m_self_);
				if (sub_index(m_self_) == 0)
				{
					carray(&m_self_, m_min_, m_max_, -1);
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

	static constexpr size_t hash(id_type s, id_type b, id_type e)
	{
		return hash_((s - b + (e - b)), (e - b)) * num_of_ele_in_cell(s)
				+ sub_index(s);
	}

	static constexpr size_t hash_(id_type const & s, id_type const & d)
	{
		//C-ORDER SLOW FIRST

		return

		(UNPACK_INDEX(s,0)% UNPACK_INDEX(d, 0))+
		(
		(UNPACK_INDEX(s,1) % UNPACK_INDEX(d, 1)) +
		(UNPACK_INDEX(s,2) % UNPACK_INDEX(d, 2)) * UNPACK_INDEX(d, 1)
		)

		* UNPACK_INDEX(d, 0);

	}
	template<size_t IFORM>
	static constexpr size_t max_hash(id_type b, id_type e)
	{
		return NProduct(unpack_index(e - b))
		* m_id_to_num_of_ele_in_cell_[sub_index_to_id<IFORM>(0)];
	}

	/**
	 *   for cut-cell
	 * @param s0
	 * @param s1
	 * @return
	 */
	static constexpr id_type out_code(id_type c,id_type s )
	{
		return out_code_(((c | FULL_OVERFLOW_FLAG)-s)&PRIMARY_ID_MASK);
	}
	static constexpr id_type out_code_(id_type c )
	{
		return
		((c>>(ID_DIGITS-1))&1UL)
		|((c>>(ID_DIGITS-3))&(4UL))
		|((c>>(ID_DIGITS*2-5))&16UL)
		|(static_cast<id_type>((c & (OVERFLOW_FLAG-1UL))!=0UL) )
		|(static_cast<id_type>((c & ((OVERFLOW_FLAG-1UL)<<ID_DIGITS))!=0UL)<<(2UL))
		|(static_cast<id_type>((c & ((OVERFLOW_FLAG-1UL)<<(ID_DIGITS*2)))!=0UL)<<(4UL))
		;
	}

	static void cut_cell(coordinates_type const & x0,coordinates_type const & x1,
	std::set<id_type>*res,id_type node_id=7, int ZAXIS = 2);

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

template<size_t N, size_t M> constexpr Real MeshIDs_<N, M>::EPSILON;

template<size_t N, size_t M> constexpr Real MeshIDs_<N, M>::_R;
template<size_t N, size_t M> constexpr typename MeshIDs_<N, M >::id_type MeshIDs_<N, M >::FULL_DIGITS;
template<size_t N, size_t M> constexpr typename MeshIDs_<N, M >::id_type MeshIDs_<N, M >::ID_DIGITS;
template<size_t N, size_t M> constexpr typename MeshIDs_<N, M >::id_type MeshIDs_<N, M >::ID_MASK;
template<size_t N, size_t M> constexpr typename MeshIDs_<N, M >::id_type MeshIDs_<N, M >::_DK;
template<size_t N, size_t M> constexpr typename MeshIDs_<N, M >::id_type MeshIDs_<N, M >::_DJ;
template<size_t N, size_t M> constexpr typename MeshIDs_<N, M >::id_type MeshIDs_<N, M >::_DI;
template<size_t N, size_t M> constexpr typename MeshIDs_<N, M >::id_type MeshIDs_<N, M >::_DA;
template<size_t N, size_t M> constexpr typename MeshIDs_<N, M >::id_type MeshIDs_<N, M >::m_id_to_index_[];
template<size_t N, size_t M> constexpr typename MeshIDs_<N, M >::id_type MeshIDs_<N, M >::m_id_to_shift_[];
template<size_t N, size_t M> constexpr typename MeshIDs_<N, M >::id_type MeshIDs_<N, M >::m_id_to_iform_[];
template<size_t N, size_t M> constexpr typename MeshIDs_<N, M >::id_type MeshIDs_<N, M >::m_sub_index_to_id_[4][3];
template<size_t N, size_t M> constexpr typename MeshIDs_<N, M >::id_type MeshIDs_<N, M >::m_id_to_num_of_ele_in_cell_[];
template<size_t N, size_t M> constexpr typename MeshIDs_<N,M >::coordinates_type MeshIDs_<N,M >::m_id_to_coordinates_shift_[ ];
template<size_t N, size_t M> constexpr int MeshIDs_<N, M>::m_vertics_num_[4][8];
template<size_t N, size_t M> constexpr typename MeshIDs_<N, M >::id_type MeshIDs_<N, M>::m_vertics_matrix_[4/* to iform*/][8/* node id*/][MAX_NUM_OF_CELL/*id shift*/];

typedef MeshIDs_<3, 4> MeshIDs;

template<size_t N, size_t M>
void MeshIDs_<N, M>::cut_cell(coordinates_type const & px0,
		coordinates_type const & px1, std::set<id_type>*res, id_type node_id,
		int ZAXIS)
{

	coordinates_type x0 = px0 - m_id_to_coordinates_shift_[node_id];

	coordinates_type x1 = px1 - m_id_to_coordinates_shift_[node_id];

	id_type face_node[3] = { 6, 5, 3 };

	for (int axe = 0; axe < 3; ++axe)
	{

		Real min = std::floor((std::min(x0[axe], x1[axe])) / (_R * 2)) * (_R * 2);
		Real max = std::floor((std::max(x0[axe], x1[axe])) / (_R * 2)) * (_R * 2);

		if (std::abs(x1[axe] - x0[axe]) < EPSILON)
		{
			continue;
		}
		else
		{
			for (Real x = min; x <= max; x += (_R * 2))
			{

				Real t = (x - x0[axe]) / (x1[axe] - x0[axe]);

				if (t < 0 || t > 1)
				{
					continue;
				}
				coordinates_type y;

				y = x0 + t * (x1 - x0);

				id_type s = (pack(y) & PRIMARY_ID_MASK)
						| m_id_to_shift_[face_node[axe]];

				res->insert(s + m_id_to_shift_[node_id]);

//				res->insert(s + ds);
//				res->insert(s - ds);

			}

		}

	};
}
}
// namespace simpla

#endif /* CORE_MESH_MESH_IDS_H_ */

