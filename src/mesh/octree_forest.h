/*
 * octree_forest.h
 *
 *  Created on: 2014年2月21日
 *      Author: salmon
 */

#ifndef OCTREE_FOREST_H_
#define OCTREE_FOREST_H_

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <limits>
#include <thread>

#include "../fetl/ntuple.h"
#include "../fetl/primitives.h"
#include "../fetl/field_rw_cache.h"
#include "../utilities/type_utilites.h"

namespace simpla
{

struct OcForest
{

	typedef OcForest this_type;
	static constexpr int MAX_NUM_NEIGHBOUR_ELEMENT = 12;
	static constexpr int MAX_NUM_VERTEX_PER_CEL = 8;
	static constexpr int NDIMS = 3;

	typedef unsigned long size_type;
	typedef unsigned long compact_index_type;
	typedef nTuple<NDIMS, Real> coordinates_type;

	//!< signed long is 63bit, unsigned long is 64 bit, add a sign bit
	static constexpr unsigned int FULL_DIGITS = std::numeric_limits<compact_index_type>::digits;

	static constexpr unsigned int D_FP_POS = 4; //!< default floating-point position

	static constexpr unsigned int INDEX_DIGITS = (FULL_DIGITS - CountBits<D_FP_POS>::n) / 3;

	static constexpr size_type INDEX_MAX = static_cast<size_type>(((1L) << (INDEX_DIGITS)) - 1);

	static constexpr size_type INDEX_MIN = 0;

	//***************************************************************************************************

	static constexpr compact_index_type NO_CARRY_FLAG = ~((1UL | (1UL << (INDEX_DIGITS * 2))
	        | (1UL << (INDEX_DIGITS * 3))) << (INDEX_DIGITS - 1));

	static constexpr compact_index_type NO_HEAD_FLAG = ~((~0UL) << (INDEX_DIGITS * 3));
	/**
	 * 	Thanks my wife Dr. CHEN Xiang Lan, for her advice on bitwise operation
	 * 	    H          m  I           m    J           m K
	 * 	|--------|--------------|--------------|-------------|
	 * 	|11111111|00000000000000|00000000000000|0000000000000| <= _MH
	 * 	|00000000|11111111111111|00000000000000|0000000000000| <= _MI
	 * 	|00000000|00000000000000|11111111111111|0000000000000| <= _MJ
	 * 	|00000000|00000000000000|00000000000000|1111111111111| <= _MK
	 *
	 * 	                    I/J/K
	 *  | INDEX_DIGITS------------------------>|
	 *  |  Root------------------->| Leaf ---->|
	 *  |11111111111111111111111111|00000000000| <=_MRI
	 *  |00000000000000000000000001|00000000000| <=_DI
	 *  |00000000000000000000000000|11111111111| <=_MTI
	 *  | Page NO.->| Tree Root  ->|
	 *  |00000000000|11111111111111|11111111111| <=_MASK
	 *
	 */
	static constexpr double dh = 1.0 / static_cast<double>(1UL << (INDEX_DIGITS + 1));
	static constexpr double idh = static_cast<double>(1UL << (INDEX_DIGITS + 1));

	static constexpr compact_index_type _DI = 1UL << (D_FP_POS + 2 * INDEX_DIGITS);
	static constexpr compact_index_type _DJ = 1UL << (D_FP_POS + INDEX_DIGITS);
	static constexpr compact_index_type _DK = 1UL << (D_FP_POS);
	static constexpr compact_index_type _DA = _DI | _DJ | _DK;

	//mask of direction
	static constexpr compact_index_type _MI = ((1UL << (INDEX_DIGITS)) - 1) << (INDEX_DIGITS * 2);
	static constexpr compact_index_type _MJ = ((1UL << (INDEX_DIGITS)) - 1) << (INDEX_DIGITS);
	static constexpr compact_index_type _MK = ((1UL << (INDEX_DIGITS)) - 1);
	static constexpr compact_index_type _MH = ((1UL << (FULL_DIGITS - INDEX_DIGITS * 3 + 1)) - 1)
	        << (INDEX_DIGITS * 3 + 1);

	// mask of sub-tree
	static constexpr compact_index_type _MTI = ((1UL << (D_FP_POS)) - 1) << (INDEX_DIGITS * 2);
	static constexpr compact_index_type _MTJ = ((1UL << (D_FP_POS)) - 1) << (INDEX_DIGITS);
	static constexpr compact_index_type _MTK = ((1UL << (D_FP_POS)) - 1);

	// mask of root
	static constexpr compact_index_type _MRI = _MI & (~_MTI);
	static constexpr compact_index_type _MRJ = _MJ & (~_MTJ);
	static constexpr compact_index_type _MRK = _MK & (~_MTK);

	nTuple<NDIMS, size_type> dims_ = { 1, 1, 1 };

	nTuple<NDIMS, size_type> strides_ = { 0, 0, 0 };

	nTuple<NDIMS, size_type> carray_digits_;

	compact_index_type _MASK;

	//***************************************************************************************************

	OcForest()
			: _MASK(NO_CARRY_FLAG)
	{

	}

	template<typename TDict>
	OcForest(TDict const & dict)
			: _MASK(NO_CARRY_FLAG)
	{
	}

	OcForest(nTuple<3, size_type> const & d)
	{
		SetDimensions(d);
	}

	~OcForest()
	{
	}

	this_type & operator=(const this_type&) = delete;

	void swap(OcForest & rhs)
	{
		std::swap(dims_, rhs.dims_);
		std::swap(_MASK, rhs._MASK);
	}

	struct index_type
	{
		compact_index_type d;

#define DEF_OP(_OP_)                                                                       \
		inline index_type & operator _OP_##=(compact_index_type r)                           \
		{                                                                                  \
			d =  ( (*this) _OP_ r).d;                                                                \
			return *this ;                                                                  \
		}                                                                                  \
		inline index_type &operator _OP_##=(index_type r)                                   \
		{                                                                                  \
			d = ( (*this) _OP_ r).d;                                                                  \
			return *this;                                                                  \
		}                                                                                  \
                                                                                           \
		inline index_type && operator _OP_(compact_index_type const &r) const                 \
		{                                                                                  \
		return 	std::move(index_type({( ((d _OP_ (r & _MI)) & _MI) |                              \
		                     ((d _OP_ (r & _MJ)) & _MJ) |                               \
		                     ((d _OP_ (r & _MK)) & _MK)                                 \
		                        )& (NO_HEAD_FLAG)}));                                         \
		}                                                                                  \
                                                                                           \
		inline index_type &&operator _OP_(index_type r) const                                \
		{                                                                                  \
			return std::move(this->operator _OP_(r.d));                                               \
		}                                                                                  \

		DEF_OP(+)
		DEF_OP(-)
		DEF_OP(^)
		DEF_OP(&)
		DEF_OP(|)
#undef DEF_OP

		bool operator==(index_type const & rhs) const
		{
			return d == rhs.d;
		}

		bool operator<(index_type const &r) const
		{
			return d < r.d;
		}
		bool operator>(index_type const &r) const
		{
			return d < r.d;
		}
	}
	;

//***************************************************************************************************

	template<typename TI>
	void SetDimensions(TI const &d, bool FORTRAN_ORDER = false)
	{
		carray_digits_[0] = D_FP_POS + 1 + ((d[0] > 0) ? (count_bits(d[0]) - 1) : 0);
		carray_digits_[1] = D_FP_POS + 1 + ((d[1] > 0) ? (count_bits(d[1]) - 1) : 0);
		carray_digits_[2] = D_FP_POS + 1 + ((d[2] > 0) ? (count_bits(d[2]) - 1) : 0);
		dims_[0] = 1UL << (carray_digits_[0] - D_FP_POS - 1);
		dims_[1] = 1UL << (carray_digits_[1] - D_FP_POS - 1);
		dims_[2] = 1UL << (carray_digits_[2] - D_FP_POS - 1);

		_MASK =

		(((1UL << (carray_digits_[0] - 1)) - 1) << (INDEX_DIGITS * 2)) |

		(((1UL << (carray_digits_[1] - 1)) - 1) << (INDEX_DIGITS)) |

		(((1UL << (carray_digits_[2] - 1)) - 1))

		;

		if (FORTRAN_ORDER)
		{
			strides_[0] = 1;
			strides_[1] = dims_[0];
			strides_[2] = dims_[1] * strides_[1];
		}
		else
		{
			strides_[2] = 1;
			strides_[1] = dims_[2];
			strides_[0] = dims_[1] * strides_[1];
		}

//		CHECK(carray_digits_);
//
//		CHECK_BIT(_DI);
//		CHECK_BIT(_MI);
//		CHECK_BIT(_MRI);
//		CHECK_BIT(_MTI);
//		CHECK_BIT(_MASK);
//
//		CHECK_BIT(_DJ);
//		CHECK_BIT(_MJ);
//		CHECK_BIT(_MRJ);
//		CHECK_BIT(_MTJ);
//		CHECK_BIT(_MASK);
//
//		CHECK_BIT(_DK);
//		CHECK_BIT(_MK);
//		CHECK_BIT(_MRK);
//		CHECK_BIT(_MTK);
//		CHECK_BIT(_MASK);

	}

	inline size_type Hash(compact_index_type d) const
	{
		d &= _MASK;
		size_type res =

		(

		(I(d) >> D_FP_POS) * strides_[0] +

		(J(d) >> D_FP_POS) * strides_[1] +

		(K(d) >> D_FP_POS) * strides_[2]

		);

		switch (_N(d))
		{
		case 1:
		case 6:
			res = ((res << 1) + res);
			break;
		case 2:
		case 5:
			res = ((res << 1) + res) + 1;
			break;
		case 4:
		case 3:
			res = ((res << 1) + res) + 2;
			break;
		}

		return res;

	}

	inline size_type Hash(index_type s) const
	{

		return std::move(Hash(s.d));

	}

	struct iterator
	{

		OcForest const & tree;

		index_type s_;

		iterator(OcForest const & m, index_type s = index_type( { 0UL }))
				: tree(m), s_(s)
		{
		}
		iterator(OcForest const & m, compact_index_type s = 0UL)
				: tree(m),

				s_(index_type( { s }))
		{
		}
		~iterator()
		{
		}

		bool operator==(iterator const & rhs) const
		{
			return s_ == rhs.s_;
		}

		bool operator!=(iterator const & rhs) const
		{
			return !(this->operator==(rhs));
		}

		index_type const & operator*() const
		{
			return s_;
		}

		size_type Hash() const
		{
			return tree.Hash(s_);
		}

		iterator & operator ++()
		{
			s_ = tree.Next(s_);
			return *this;
		}

		index_type * operator ->()
		{
			return &s_;
		}
		index_type const* operator ->() const
		{
			return &s_;
		}
	};

	/**
	 *
	 * @param total
	 * @param n
	 * @return
	 */
	iterator begin(int IFORM, int total = 1, int sub = 0) const
	{
		compact_index_type s = ((dims_[0] * (sub) / total) << (INDEX_DIGITS * 2 + D_FP_POS - 1)) & _MASK;

		if (IFORM == EDGE)
		{
			s |= (_DI >> 1);
		}
		else if (IFORM == FACE)
		{
			s |= ((_DI | _DJ) >> 1);
		}
		else if (IFORM == VOLUME)
		{
			s |= ((_DI | _DJ | _DK) >> 1);
		}
		return iterator(*this, s);
	}

	iterator end(int IFORM, int total = 1, int sub = 0) const
	{
		iterator res = begin(IFORM);
		res->d += ((dims_[0] / total) << (INDEX_DIGITS * 2 + D_FP_POS));
		return res;
	}

	compact_index_type Next(compact_index_type s) const
	{

		auto n = _N(s);

		if (n == 0 || n == 4 || n == 3 || n == 7)
		{
			s += _DK | _DJ | _DI;

			auto m = (((~s) & (1UL << (carray_digits_[2] - 1))) << (INDEX_DIGITS + D_FP_POS + 1 - carray_digits_[2])
			        | ((~s) & (1UL << (carray_digits_[1] - 1 + INDEX_DIGITS)))
			                << (INDEX_DIGITS + D_FP_POS + 1 - carray_digits_[1]));

			auto mm =
			        (~((s & (1UL << (carray_digits_[2] - 1))) | (s & (1UL << (carray_digits_[1] - 1 + INDEX_DIGITS)))));

			s = (s - m) & mm;
		}

		s = _R(s);

		return s;
	}

	index_type Next(index_type s) const
	{
		return index_type( { Next(s.d) });
	}

	//***************************************************************************************************
	//  Traversal

	template<int IFORM, typename TF, typename ...Args>
	void Traversal(TF &&fun, Args && ...args) const
	{
		auto it = this->begin(IFORM), ie = this->end(IFORM);

		while (true)
		{
			fun(*it, args...);

			++it;
			if (it->d == ie->d)
				break;
		}

	}

	template<int IFORM, typename TF, typename ... Args>
	void ParallelTraversal(TF &&fun, Args && ...args) const
	{
		const unsigned int num_threads = std::thread::hardware_concurrency();

		std::vector<std::thread> threads;

		for (unsigned int thread_id = 0; thread_id < num_threads; ++thread_id)
		{
			auto ib = this->begin(IFORM, num_threads, thread_id);
			auto ie = this->end(IFORM, num_threads, thread_id);

			threads.emplace_back(

			std::thread(

			[ib,ie](TF fun2, Args ... args2 )
			{
				for (auto it =ib; it != ie; ++it)
				{
					fun2(*it,args2...);
				}

			}, fun, std::forward<Args >(args)...

			));
		}

		for (auto & t : threads)
		{
			t.join();
		}
	}

	template<int IFORM, typename TF, typename ... Args>
	void ParallelCachedTraversal(TF &&fun, Args && ...args) const
	{
		const unsigned int num_threads = std::thread::hardware_concurrency();

		std::vector<std::thread> threads;

		for (unsigned int thread_id = 0; thread_id < num_threads; ++thread_id)
		{
			auto ib = this->begin(IFORM, num_threads, thread_id);
			auto ie = this->end(IFORM, num_threads, thread_id);

			threads.emplace_back(

			std::thread(

			[ib,ie](TF fun2,typename Cache<Args>::type && ... args2 )
			{
				for (auto it =ib; it != ie; ++it)
				{
					RefreshCache(*it,args2...);

					fun2(*it,args2...);

					FlushCache(*it,args2...);
				}

			}, fun, typename Cache<Args >::type(args)...

			)

			);
		}

		for (auto & t : threads)
		{
			t.join();
		}
	}

//***************************************************************************************************

	nTuple<3, size_type>
	const & GetDimensions() const
	{
		return dims_;
	}
	nTuple<3, Real> GetExtent() const
	{
		return nTuple<3, Real>( {

		(dims_[0] << D_FP_POS) * dh,

		(dims_[1] << D_FP_POS) * dh,

		(dims_[2] << D_FP_POS) * dh

		});
	}
	size_type GetNumOfElements(int IFORM = VERTEX) const
	{
		return dims_[0] * dims_[1] * dims_[2] * ((IFORM == VERTEX || IFORM == VOLUME) ? 1 : 3);
	}

	inline index_type GetIndex(nTuple<3, Real> const & x, unsigned long h = 0) const
	{
		return index_type( {

		(

		(h << (INDEX_DIGITS * 3)) |

		(static_cast<size_type>(std::floor(x[0] * idh)) << (INDEX_DIGITS * 2)) |

		(static_cast<size_type>(std::floor(x[1] * idh)) << (INDEX_DIGITS)) |

		static_cast<size_type>(std::floor(x[2] * idh))

		) & _MASK

		});

	}

	inline nTuple<3, Real> GetCoordinates(index_type s) const
	{
		s &= _MASK;

		return nTuple<3, Real>( {

		static_cast<Real>(I(s)) * dh,

		static_cast<Real>(J(s)) * dh,

		static_cast<Real>(K(s)) * dh

		});

	}

//***************************************************************************************************
//* Auxiliary functions
//***************************************************************************************************

	size_type H(compact_index_type s) const
	{
		return s >> (INDEX_DIGITS * 3);
	}

	size_type H(index_type s) const
	{
		return H(s.d);
	}

	size_type I(compact_index_type s) const
	{
		return (s & _MI) >> (INDEX_DIGITS * 2);
	}

	size_type I(index_type s) const
	{
		return I(s.d);
	}

	size_type J(compact_index_type s) const
	{
		return (s & _MJ) >> (INDEX_DIGITS);
	}

	size_type J(index_type s) const
	{
		return H(s.d);
	}
	size_type K(compact_index_type s) const
	{
		return (s & _MK);
	}

	size_type K(index_type s) const
	{
		return K(s.d);
	}

	/**
	 *  rotate vector direction  mask
	 *  (1/2,0,0) => (0,1/2,0) or   (1/2,1/2,0) => (0,1/2,1/2)
	 * @param s
	 * @return
	 */
	index_type _R(index_type s) const
	{
		s.d = _R(s.d);
		return s;
	}

	compact_index_type _R(compact_index_type s) const
	{
		compact_index_type r = s;

		r &= ~(_DA >> (H(s) + 1));

		r |= ((s & (_DI >> (H(s) + 1))) >> INDEX_DIGITS) |

		((s & (_DJ >> (H(s) + 1))) >> INDEX_DIGITS) |

		((s & (_DK >> (H(s) + 1))) << (INDEX_DIGITS * 2))

		;
		return r;
	}

	/**
	 *  rotate vector direction  mask
	 *  (1/2,0,0) => (0,0,1/2) or   (1/2,1/2,0) => (1/2,0,1/2)
	 * @param s
	 * @return
	 */
	index_type _RR(index_type s) const
	{
		s.d = _RR(s.d);
		return s;
	}

	compact_index_type _RR(compact_index_type s) const
	{
		compact_index_type r = s;
		r &= ~(_DA >> (H(s) + 1));

		r |= ((s & (_DI >> (H(s) + 1))) >> (INDEX_DIGITS * 2)) |

		((s & (_DJ >> (H(s) + 1))) << INDEX_DIGITS) |

		((s & (_DK >> (H(s) + 1))) << INDEX_DIGITS)

		;
		return r;
	}

	/**
	 *    (1/2,0,1/2) => (0,1/2,0) or   (1/2,0,0) => (0,1/2,1/2)
	 * @param s
	 * @return
	 */
	index_type _I(index_type s) const
	{
		s.d = _I(s.d);
		return s;
	}

	compact_index_type _I(compact_index_type s) const
	{
		return std::move((s & (~(_DA >> (H(s) + 1)))) | ((~(s & (_DA >> (H(s) + 1)))) & (_DA >> (H(s) + 1))));
	}

//! get the direction of vector(edge) 0=>x 1=>y 2=>z
	size_type _N(compact_index_type s) const
	{

		s = (s & (_DA >> (H(s) + 1))) >> (D_FP_POS - H(s) - 1);
		return

		(

		(s >> (INDEX_DIGITS * 2 - 2)) |

		(s >> (INDEX_DIGITS - 1)) |

		s

		) & (7UL)

		;
	}
	size_type _N(index_type s) const
	{
		return std::move(_N(s.d));
	}

	size_type _C(compact_index_type s) const
	{
		size_type res = 0;
		switch (_N(s))
		{
		case 1:
		case 6:
			res = 0;
			break;
		case 2:
		case 5:
			res = 1;
			break;
		case 4:
		case 3:
			res = 2;
			break;
		}
		return res;
	}
	size_type _C(index_type s) const
	{
		return std::move(_C(s.d));
	}
	index_type _D(index_type s) const
	{
		s.d = _D(s.d);
		return s;
	}
	compact_index_type _D(compact_index_type s) const
	{
		return s & (_DA >> (H(s) + 1));

	}

	template<int I>
	inline int GetAdjacentCells(Int2Type<I>, Int2Type<I>, index_type s, index_type *v) const
	{
		v[0] = s;
		return 1;
	}

	inline int GetAdjacentCells(Int2Type<EDGE>, Int2Type<VERTEX>, index_type s, index_type *v) const
	{
		v[0] = s + _D(s);
		v[1] = s - _D(s);
		return 2;
	}

	inline int GetAdjacentCells(Int2Type<FACE>, Int2Type<VERTEX>, index_type s, index_type *v) const
	{
		/**
		 *
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
		 *
		 *
		 */

		auto di = _D(_R(_I(s)));
		auto dj = _D(_RR(_I(s)));

		v[0] = s - di - dj;
		v[1] = s - di - dj;
		v[2] = s + di + dj;
		v[3] = s + di + dj;

		return 4;
	}

	inline int GetAdjacentCells(Int2Type<VOLUME>, Int2Type<VERTEX>, index_type const &s, index_type *v) const
	{
		/**
		 *
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
		 *
		 */
		auto di = _DI >> (H(s) + 1);
		auto dj = _DJ >> (H(s) + 1);
		auto dk = _DK >> (H(s) + 1);

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

	inline int GetAdjacentCells(Int2Type<VERTEX>, Int2Type<EDGE>, index_type s, index_type *v) const
	{
		/**
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
		 *
		 */

		auto di = _DI >> (H(s) + 1);
		auto dj = _DJ >> (H(s) + 1);
		auto dk = _DK >> (H(s) + 1);

		v[0] = s + di;
		v[1] = s - di;

		v[2] = s + dj;
		v[3] = s - dj;

		v[4] = s + dk;
		v[5] = s - dk;

		return 6;
	}

	inline int GetAdjacentCells(Int2Type<FACE>, Int2Type<EDGE>, index_type s, index_type *v) const
	{

		/**
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
		 *        0---------------1   ---> x
		 *
		 *
		 */
		auto d1 = _D(_R(_I(s)));
		auto d2 = _D(_RR(_I(s)));
		v[0] = s - d1;
		v[1] = s + d1;
		v[2] = s - d2;
		v[3] = s + d2;

		return 4;
	}

	inline int GetAdjacentCells(Int2Type<VOLUME>, Int2Type<EDGE>, index_type s, index_type *v) const
	{

		/**
		 *
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
		 *
		 */
		auto di = _DI >> (H(s) + 1);
		auto dj = _DJ >> (H(s) + 1);
		auto dk = _DK >> (H(s) + 1);

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

	inline int GetAdjacentCells(Int2Type<VERTEX>, Int2Type<FACE>, index_type s, index_type *v) const
	{
		/**
		 *
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
		 *
		 */
		auto di = _DI >> (H(s) + 1);
		auto dj = _DJ >> (H(s) + 1);
		auto dk = _DK >> (H(s) + 1);

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

	inline int GetAdjacentCells(Int2Type<EDGE>, Int2Type<FACE>, index_type s, index_type *v) const
	{

		/**
		 *
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
		 *
		 */

		auto d1 = _D(_R(s));
		auto d2 = _D(_RR(s));

		v[0] = s - d1;
		v[1] = s + d1;
		v[2] = s - d2;
		v[3] = s + d2;

		return 4;
	}

	inline int GetAdjacentCells(Int2Type<VOLUME>, Int2Type<FACE>, index_type s, index_type *v) const
	{

		/**
		 *
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
		 *
		 */

		auto di = _DI >> (H(s) + 1);
		auto dj = _DJ >> (H(s) + 1);
		auto dk = _DK >> (H(s) + 1);

		v[0] = s - di;
		v[1] = s + di;

		v[2] = s - di;
		v[3] = s + dj;

		v[4] = s - dk;
		v[5] = s + dk;

		return 6;
	}

	inline int GetAdjacentCells(Int2Type<VERTEX>, Int2Type<VOLUME>, index_type s, index_type *v) const
	{
		/**
		 *
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
		 *
		 */

		auto di = _DI >> (H(s) + 1);
		auto dj = _DJ >> (H(s) + 1);
		auto dk = _DK >> (H(s) + 1);

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

	inline int GetAdjacentCells(Int2Type<EDGE>, Int2Type<VOLUME>, index_type s, index_type *v) const
	{

		/**
		 *
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
		 *
		 */

		auto d1 = _D(_R(s));
		auto d2 = _D(_RR(s));

		v[0] = s - d1 - d2;
		v[1] = s + d1 - d2;
		v[2] = s - d1 + d2;
		v[3] = s + d1 + d2;
		return 4;
	}

	inline int GetAdjacentCells(Int2Type<FACE>, Int2Type<VOLUME>, index_type s, index_type *v) const
	{

		/**
		 *
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
		 *
		 */

		auto d = _D(_I(s));
		v[0] = s + d;
		v[1] = s - d;

		return 2;
	}

}
;

inline unsigned long make_hash(OcForest::iterator s)
{
	return s.Hash();
}
}
// namespace simpla

#endif /* OCTREE_FOREST_H_ */
