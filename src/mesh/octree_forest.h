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
#include "../utilities/memory_pool.h"

namespace simpla
{

struct OcForest
{

	typedef OcForest this_type;
	static constexpr int MAX_NUM_VERTEX_PER_CEL = 8;
	static constexpr int NUM_OF_DIMS = 3;

	typedef unsigned long size_type;
	typedef unsigned long compact_index_type;

	//!< signed long is 63bit, unsigned long is 64 bit, add a sign bit
	static constexpr unsigned int FULL_DIGITS = std::numeric_limits<compact_index_type>::digits;

	static constexpr unsigned int D_FP_POS = 4; //!< default floating-point position

	static constexpr unsigned int INDEX_DIGITS = (FULL_DIGITS - CountBits<D_FP_POS>::n) / 3;

	static constexpr size_type INDEX_MAX = static_cast<size_type>(((1L) << (INDEX_DIGITS)) - 1);

	static constexpr size_type INDEX_MIN = 0;

	static constexpr double dh = 1.0 / static_cast<double>(INDEX_MAX + 1);

	OcForest()
	{
	}
	template<typename TDict>
	OcForest(TDict const & dict)
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

	//***************************************************************************************************

	static constexpr compact_index_type NO_CARRY_FLAG = ~((1UL | (1UL << (INDEX_DIGITS * 2))
	        | (1UL << (INDEX_DIGITS * 3))) << (INDEX_DIGITS - 1));

	static constexpr compact_index_type NO_HEAD_FLAG = (~0UL) << (INDEX_DIGITS * 3);
	/**
	 * 	Thanks my wife Dr. CHEN Xiang Lan, for her advice on  these bitwise operation
	 * 	               m            m             m
	 * 	|--------|------------|--------------|-------------|
	 * 	     H         I              J             K
	 */
	struct index_type
	{
		compact_index_type d;

#define DEF_OP(_OP_)                                                                       \
		inline index_type operator _OP_##=(compact_index_type r)                           \
		{                                                                                  \
			d =  ( (*this) _OP_ r).d;                                                                \
			return *this;                                                                  \
		}                                                                                  \
		inline index_type operator _OP_##=(index_type r)                                   \
		{                                                                                  \
			d = ( (*this) _OP_ r).d;                                                                  \
			return *this;                                                                  \
		}                                                                                  \
                                                                                           \
		inline index_type operator _OP_(compact_index_type const &r) const                 \
		{                                                                                  \
			return index_type( { (d & NO_HEAD_FLAG) | ((d _OP_ r) & NO_CARRY_FLAG) });     \
		}                                                                                  \
                                                                                           \
		inline index_type operator _OP_(index_type r) const                                \
		{                                                                                  \
			return this->operator _OP_(r.d);                                               \
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

		size_type H() const
		{
			return d >> (INDEX_DIGITS * 3);
		}

		size_type I() const
		{
			return (d & _MI) >> (INDEX_DIGITS * 2);
		}
		size_type J() const
		{
			return (d & _MJ) >> (INDEX_DIGITS);
		}
		size_type K() const
		{
			return (d & _MK);
		}

	}
	;

	static constexpr compact_index_type _DI = 1UL << (D_FP_POS + 2 * INDEX_DIGITS);
	static constexpr compact_index_type _DJ = 1UL << (D_FP_POS + INDEX_DIGITS);
	static constexpr compact_index_type _DK = 1UL << (D_FP_POS);
	static constexpr compact_index_type _DA = _DI | _DJ | _DK;

	//mask of direction
	static constexpr compact_index_type _MI = ((1UL << (INDEX_DIGITS + 1)) - 1) << (INDEX_DIGITS * 2);
	static constexpr compact_index_type _MJ = ((1UL << (INDEX_DIGITS + 1)) - 1) << (INDEX_DIGITS);
	static constexpr compact_index_type _MK = ((1UL << (INDEX_DIGITS + 1)) - 1);
	static constexpr compact_index_type _MH = ((1UL << (FULL_DIGITS - INDEX_DIGITS * 3 + 1)) - 1) << (INDEX_DIGITS * 3);

	// mask of sub-tree
	static constexpr compact_index_type _MTI = ((1UL << (D_FP_POS + 1)) - 1) << (INDEX_DIGITS * 2);
	static constexpr compact_index_type _MTJ = ((1UL << (D_FP_POS + 1)) - 1) << (INDEX_DIGITS);
	static constexpr compact_index_type _MTK = ((1UL << (D_FP_POS + 1)) - 1);

	// mask of root
	static constexpr compact_index_type _MRI = _MI & (~_MTI);
	static constexpr compact_index_type _MRJ = _MJ & (~_MTJ);
	static constexpr compact_index_type _MRK = _MK & (~_MTK);

//***************************************************************************************************

	nTuple<3, size_type> dims_ = { 1, 1, 1 };

	nTuple<NUM_OF_DIMS, size_type> strides_ = { 0, 0, 0 };

	compact_index_type _MASK = NO_CARRY_FLAG;

	template<typename TI>
	void SetDimensions(TI const &d)
	{
		dims_[0] = 1UL << count_bits(d[0]);
		dims_[1] = 1UL << count_bits(d[1]);
		dims_[2] = 1UL << count_bits(d[2]);

		_MASK =

		(((1UL << (count_bits(d[0]) + D_FP_POS + 1)) - 1) << (INDEX_DIGITS * 2)) |

		(((1UL << (count_bits(d[1]) + D_FP_POS + 1)) - 1) << (INDEX_DIGITS)) |

		(((1UL << (count_bits(d[2]) + D_FP_POS + 1)) - 1))

		;
		CHECK_BIT(_MASK);

	}

	template<int IFORM>
	inline size_type Hash(index_type s) const
	{
		s &= _MASK;

		return (

		(s.I() >> (D_FP_POS)) * strides_[0] +

		(s.J() >> (D_FP_POS)) * strides_[1] +

		(s.K() >> (D_FP_POS)) * strides_[2]

		);

	}

	template<int IFORM>
	struct iterator
	{

		static constexpr int IForm = IFORM;

		OcForest const & tree;

		index_type s_;

		iterator(OcForest const & m, index_type s = index_type( { 0UL }))
				: tree(m), s_(s)
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
			return tree.Hash<IForm>(s_);
		}

		iterator & operator ++()
		{
			s_ = tree.Next<IForm>(s_);
			return *this;
		}

	};

	template<int IFORM> iterator<IFORM> begin(int total = 1, int sub = 0) const
	{
		auto dims_ = GetDimensions();

		index_type s = { 0 };

//		if (IFORM == EDGE)
//		{
//			s |= (_DI >> (s.H() + 1));
//		}
//		else if (IFORM == FACE)
//		{
//			s |= ((_DJ | _DK) >> (s.H() + 1));
//		}
		return iterator<IFORM>(*this, s);
	}

	template<int IFORM> iterator<IFORM> end(int total = 1, int sub = 0) const
	{

		auto dims_ = GetDimensions();

		index_type s = { 0 };

		if (IFORM == EDGE)
		{
			s += _DI >> 1;
		}
		if (IFORM == FACE)
		{
			s += (_DJ | _DK) >> 1;
		}
		return iterator<IFORM>(*this, s);
	}

	template<int IForm>
	index_type Next(index_type s) const
	{

		if (IForm == VERTEX || IForm == VOLUME)
		{
			s += _DK;

			if (s & _MRK == 0)
			{
				s += _DJ;
			}
			if (s & _MRJ == 0)
			{
				s += _DI;
			}
		}

		return s;
	}

//***************************************************************************************************
//  Traversal

	template<int IFORM, typename TF, typename ... Args>
	void ParallelTraversal(TF &&fun, Args && ...args) const
	{
		const unsigned int num_threads = std::thread::hardware_concurrency();

		std::vector<std::thread> threads;

		for (unsigned int thread_id = 0; thread_id < num_threads; ++thread_id)
		{
			auto ib = this->begin<IFORM>(num_threads, thread_id);
			auto ie = this->end<IFORM>(num_threads, thread_id);

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

	template<int IFORM, typename TF, typename ...Args>
	void Traversal(TF &&fun, Args && ...args) const
	{
		auto it = this->begin<IFORM>(), ie = this->end<IFORM>();
		for (; it != ie; ++it)
		{

			fun(*it, std::forward<Args>(args)...);
		}
	}

	template<int IFORM, typename TF, typename ... Args>
	void ParallelCachedTraversal(TF &&fun, Args && ...args) const
	{
		const unsigned int num_threads = std::thread::hardware_concurrency();

		std::vector<std::thread> threads;

		for (unsigned int thread_id = 0; thread_id < num_threads; ++thread_id)
		{
			auto ib = this->begin<IFORM>(num_threads, thread_id);
			auto ie = this->end<IFORM>(num_threads, thread_id);

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

	nTuple<3, size_type> const & GetDimensions() const
	{
		return dims_;

	}

	size_type GetNumOfElements(int IFORM = VERTEX) const
	{
		auto dims = GetDimensions();
		return dims[0] * dims[1] * dims[2] * ((IFORM == VERTEX || IFORM == VOLUME) ? 1 : 3);
	}
	nTuple<3, Real> GetDx() const
	{
		return nTuple<3, Real>( {

		static_cast<Real>(1U << (D_FP_POS)) * dh,

		static_cast<Real>(1U << (D_FP_POS)) * dh,

		static_cast<Real>(1U << (D_FP_POS)) * dh });
	}

	inline index_type GetIndex(nTuple<3, Real> const & x, unsigned int H = 0) const
	{
		index_type res;

		ASSERT(0 <= x[0] && x[0] <= 1.0);

		ASSERT(0 <= x[1] && x[1] <= 1.0);

		ASSERT(0 <= x[2] && x[2] <= 1.0);

//		res.H() = H;
//
//		res.I = static_cast<size_type>(std::floor(x[0] * static_cast<Real>(INDEX_MAX + 1)))
//		        & ((~0UL) << (D_FP_POS - H));
//
//		res.J = static_cast<size_type>(std::floor(x[1] * static_cast<Real>(INDEX_MAX + 1)))
//		        & ((~0UL) << (D_FP_POS - H));
//
//		res.K = static_cast<size_type>(std::floor(x[2] * static_cast<Real>(INDEX_MAX + 1)))
//		        & ((~0UL) << (D_FP_POS - H));

		return std::move(res);
	}

	inline nTuple<3, Real> GetCoordinates(index_type s) const
	{
		s &= _MASK;

		return nTuple<3, Real>( {

		static_cast<Real>(s.I()) * dh,

		static_cast<Real>(s.J()) * dh,

		static_cast<Real>(s.K()) * dh

		});

	}

//***************************************************************************************************
//* Auxiliary functions
//***************************************************************************************************

	/**
	 *  rotate vector direction  mask
	 *  (1/2,0,0) => (0,1/2,0) or   (1/2,1/2,0) => (0,1/2,1/2)
	 * @param s
	 * @return
	 */
	index_type _R(index_type s) const
	{
//		index_type f = _D(s);
//		s.I = ~0UL;
//		s.K = ~0UL;
//		_C(s) &= ~(_MA >> (s.H() + 1));
//		CHECK ( s.H());
//		CHECK_BIT(~(_MA >> (s.H() + 1)));
//		CHECK_BIT(_C(s));
//		s.I |= f.K;
//		s.J |= f.I;
//		s.K |= f.J;
//
//		CHECK_BIT(_C(f));
		return s;
	}

	compact_index_type _R(compact_index_type s) const
	{
//		return std::move(_C(_R(_C(s))));
	}

	/**
	 *  rotate vector direction  mask
	 *  (1/2,0,0) => (0,0,1/2) or   (1/2,1/2,0) => (1/2,0,1/2)
	 * @param s
	 * @return
	 */
	index_type _RR(index_type s) const
	{
//		index_type f = s & (_DA >> (s.H() + 1));
//		_C(s) &= ~(_DA >> (s.H() + 1));
//		s.I |= f.J;
//		s.J |= f.K;
//		s.K |= f.I;
		return s;
	}

	compact_index_type _RR(compact_index_type s) const
	{
		return std::move(_RR(s));
	}

	compact_index_type _I(compact_index_type s) const
	{
//		return (s & ~(_MA >> (_C(s).H + 1))) | (~(s & (_DA >> (_C(s).H + 1))));
	}

	index_type _I(index_type s) const
	{
//		return std::move(_C(_I(_C(s))));
	}

//! get the direction of vector(edge) 0=>x 1=>y 2=>z
	size_type _N(index_type s) const
	{
//		return ((s.J >> (D_FP_POS - s.H() - 1)) & 1UL) |
//
//		((s.K >> (D_FP_POS - s.H() - 2)) & 2UL);
	}
	size_type _N(compact_index_type s) const
	{
//		return std::move(_N(_C(s)));
	}

	index_type _D(index_type s) const
	{
//		return s & (_DA >> (s.H() + 1));
	}
	compact_index_type _D(compact_index_type s) const
	{
//		return _C(_D(_C(s)));
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
		auto di = _DI >> (s.H() + 1);
		auto dj = _DJ >> (s.H() + 1);
		auto dk = _DK >> (s.H() + 1);

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

		auto di = _DI >> (s.H() + 1);
		auto dj = _DJ >> (s.H() + 1);
		auto dk = _DK >> (s.H() + 1);

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
		auto di = _DI >> (s.H() + 1);
		auto dj = _DJ >> (s.H() + 1);
		auto dk = _DK >> (s.H() + 1);

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
		auto di = _DI >> (s.H() + 1);
		auto dj = _DJ >> (s.H() + 1);
		auto dk = _DK >> (s.H() + 1);

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

		auto di = _DI >> (s.H() + 1);
		auto dj = _DJ >> (s.H() + 1);
		auto dk = _DK >> (s.H() + 1);

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

		auto di = _DI >> (s.H() + 1);
		auto dj = _DJ >> (s.H() + 1);
		auto dk = _DK >> (s.H() + 1);

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

}
// namespace simpla

#endif /* OCTREE_FOREST_H_ */
