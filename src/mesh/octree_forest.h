/*
 * octree_forest.h
 *
 *  Created on: 2014年2月21日
 *      Author: salmon
 */

#ifndef OCTREE_FOREST_H_
#define OCTREE_FOREST_H_

#include <algorithm>
#include <cmath>
#include <limits>

#include "../fetl/ntuple.h"
#include "../utilities/type_utilites.h"

namespace simpla
{

struct OcForest
{

	typedef OcForest this_type;
	static constexpr int MAX_NUM_VERTEX_PER_CEL = 8;
	static constexpr int NUM_OF_DIMS = 3;

	typedef unsigned long size_type;
	typedef unsigned long compact_index_type;

	/**
	 * 	Thanks my wife Dr. CHEN Xiang Lan, for her advice on  these bitwise operation
	 * 	               m            m             m
	 * 	|--------|------------|--------------|-------------|
	 * 	               I              J             K
	 */
	//!< signed long is 63bit, unsigned long is 64 bit, add a sign bit
	static constexpr unsigned int FULL_DIGITS = std::numeric_limits<compact_index_type>::digits;

	static constexpr unsigned int MAX_TREE_HEIGHT = 4;

	static constexpr unsigned int INDEX_DIGITS = (FULL_DIGITS - CountBits<MAX_TREE_HEIGHT>::n) / 3;

	static constexpr unsigned int DIGITS_HEAD = FULL_DIGITS - INDEX_DIGITS * 3;

	static constexpr size_type INDEX_MAX = static_cast<size_type>(((1L) << (INDEX_DIGITS)) - 1);

	static constexpr size_type INDEX_MIN = 0;

	static constexpr double dh = 1.0 / static_cast<double>(INDEX_MAX + 1);

	struct index_type
	{
		size_type H :DIGITS_HEAD;
		size_type I :INDEX_DIGITS;
		size_type J :INDEX_DIGITS;
		size_type K :INDEX_DIGITS;

#define DEF_OP(_OP_)                                            \
		index_type operator _OP_##=(index_type const &r)        \
		{                                                       \
			H = std::max(H, r.H);                               \
			I _OP_##= r.I;                                      \
			J _OP_##= r.J;                                      \
			K _OP_##= r.K;                                      \
			return *this;                                       \
		}                                                       \
                                                                \
		index_type operator _OP_ (index_type const &r)const          \
		{                                                       \
			index_type t;                                       \
			t.H = std::max(H, r.H);                             \
			t.I = I _OP_ r.I;                                   \
			t.J = J _OP_ r.J;                                   \
			t.K = K _OP_ r.K;                                   \
			return t;                                           \
		}                                                       \
		index_type operator _OP_##=(compact_index_type  r ) \
		{    this->operator _OP_##=(_C(r));  return *this;      } \
																\
		index_type operator _OP_ (compact_index_type  r)const  \
		{   return std::move(this->operator _OP_ (_C(r)));    } \

		DEF_OP(+)
		DEF_OP(-)
		DEF_OP(^)
		DEF_OP(&)
		DEF_OP(|)
#undef DEF_OP

	};

	nTuple<3, unsigned int> index_digits_ = { INDEX_DIGITS - MAX_TREE_HEIGHT, INDEX_DIGITS - MAX_TREE_HEIGHT,
	        INDEX_DIGITS - MAX_TREE_HEIGHT };

	compact_index_type _MI = 0UL;
	compact_index_type _MJ = 0UL;
	compact_index_type _MK = 0UL;
	compact_index_type _MA = _MI | _MJ | _MK;

	//  public:
	OcForest()
	{
	}
	OcForest(nTuple<3, unsigned int> const & d)
	{
		SetDimensions(d);
		Update();
	}

	~OcForest()
	{
	}
	this_type & operator=(const this_type&) = delete;

	static compact_index_type &_C(index_type &s)
	{
		return *reinterpret_cast<compact_index_type *>(&s);
	}

	static compact_index_type const &_C(index_type const &s)
	{
		return *reinterpret_cast<compact_index_type const*>(&s);
	}

	void SetDimensions(nTuple<3, unsigned int> const & d)
	{
		index_digits_[0] = count_bits(d[0]) - 1;
		index_digits_[1] = count_bits(d[1]) - 1;
		index_digits_[2] = count_bits(d[2]) - 1;
		Update();
	}
	nTuple<3, unsigned int> GetDimensions() const
	{
		return nTuple<3, unsigned int>( { 1U << index_digits_[0], 1U << index_digits_[1], 1U << index_digits_[2] });

	}
	void Update()
	{
		_MI = _C(index_type( { 0, 1U << (INDEX_DIGITS - index_digits_[0] - 1), 0, 0 }));
		_MJ = _C(index_type( { 0, 0, 1U << (INDEX_DIGITS - index_digits_[1] - 1), 0 }));
		_MK = _C(index_type( { 0, 0, 0, 1U << (INDEX_DIGITS - index_digits_[2] - 1) }));
		_MA = _MI | _MJ | _MK;
	}

	static index_type &_C(compact_index_type &s)
	{
		return *reinterpret_cast<index_type *>(&s);
	}

	static index_type const &_C(compact_index_type const &s)
	{
		return *reinterpret_cast<index_type const*>(&s);
	}

	inline size_type HashRootIndex(index_type s, size_type const strides[3]) const
	{

		return (

		(s.I >> (INDEX_DIGITS - index_digits_[0])) * strides[0] +

		(s.J >> (INDEX_DIGITS - index_digits_[1])) * strides[1] +

		(s.K >> (INDEX_DIGITS - index_digits_[2])) * strides[2]

		);

	}

	inline index_type GetIndex(nTuple<3, Real> const & x, unsigned int H = 0) const
	{
		index_type res;

		ASSERT(0<=x[0] && x[0]<=1.0);

		ASSERT(0<=x[1] && x[1]<=1.0);

		ASSERT(0<=x[2] && x[2]<=1.0);

		res.H = H;

		res.I = static_cast<size_type>(std::floor(x[0] * static_cast<Real>(INDEX_MAX + 1)))
		        & ((~0UL) << (INDEX_DIGITS - index_digits_[0]));

		res.J = static_cast<size_type>(std::floor(x[1] * static_cast<Real>(INDEX_MAX + 1)))
		        & ((~0UL) << (INDEX_DIGITS - index_digits_[1]));

		res.K = static_cast<size_type>(std::floor(x[2] * static_cast<Real>(INDEX_MAX + 1)))
		        & ((~0UL) << (INDEX_DIGITS - index_digits_[2]));

		return std::move(res);
	}

	inline nTuple<3, Real> GetCoordinates(index_type const & s) const
	{

		return nTuple<3, Real>( {

		static_cast<Real>(s.I) * dh,

		static_cast<Real>(s.J) * dh,

		static_cast<Real>(s.K) * dh

		});

	}

	index_type _II(index_type const &s)
	{
		index_type t = s;
		t.I += 1UL << (INDEX_DIGITS - index_digits_[0]);
		return t;
	}
	index_type _IJ(index_type const &s)
	{
		index_type t = s;
		t.J += 1UL << (INDEX_DIGITS - index_digits_[1]);
		return t;
	}
	index_type _IK(index_type const &s)
	{
		index_type t = s;
		t.K += 1UL << (INDEX_DIGITS - index_digits_[2]);
		return t;
	}

	index_type _DI(index_type const &s)
	{
		index_type t = s;
		t.I -= 1UL << (INDEX_DIGITS - index_digits_[0]);
		return t;
	}
	index_type _DJ(index_type const &s)
	{
		index_type t = s;
		t.J -= 1UL << (INDEX_DIGITS - index_digits_[1]);
		return t;
	}
	index_type _DK(index_type const &s)
	{
		index_type t = s;
		t.K += 1UL << (INDEX_DIGITS - index_digits_[2]);
		return t;
	}
//***************************************************************************************************

	template<int I>
	inline int GetAdjacentCells(Int2Type<I>, Int2Type<I>, index_type const & s, index_type *v) const
	{
		v[0] = s;
		return 1;
	}

	inline int GetAdjacentCells(Int2Type<EDGE>, Int2Type<VERTEX>, index_type s, index_type *v) const
	{
		v[0] = s + s & (_MA >> s.H);
		v[1] = s - s & (_MA >> s.H);
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

		v[0] = s - (_MA >> s.H);
		v[1] = s + ((_MI | _MJ) >> s.H);
		v[2] = s + ((_MJ | _MK) >> s.H);
		v[3] = s + ((_MK | _MI) >> s.H);

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

		v[0] = s - (_MK >> s.H);
		v[1] = s + (_MI >> s.H);
		v[2] = s + (_MJ >> s.H);
		v[3] = s + ((_MI | _MJ) >> s.H);

//			v[4] = (INC(2)) + s;
//			v[5] = (INC(2) | INC(0)) + s;
//			v[6] = (INC(2) | INC(1) | INC(1)) + s;
//			v[7] = (INC(2) | INC(1)) + s;

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

		v[0] = s | (_MI >> s.H);
		v[1] = s | (_MJ >> s.H);
		v[2] = s | (_MK >> s.H);
		v[3] = v[0];
		v[3].I -= 1UL << (INDEX_DIGITS - index_digits_[0]);
		v[4] = v[1];
		v[4].J -= 1UL << (INDEX_DIGITS - index_digits_[1]);
		v[5] = v[2];
		v[5].K -= 1UL << (INDEX_DIGITS - index_digits_[2]);

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

		v[0] = s - s & ((_MI | _MJ) >> s.H);
		v[1] = s + s & ((_MI | _MJ) >> s.H);
		v[2] = s - s & ((_MK | _MJ) >> s.H);
		v[3] = s + s & ((_MK | _MJ) >> s.H);
		return 4;
	}
//
//	inline int GetAdjacentCells(Int2Type<VOLUME>, Int2Type<EDGE>, index_type s, index_type *v) const
//	{
//
//		/**
//		 *
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
//		 *
//		 */
//
//		if (v != nullptr)
//		{
//			v[0] = s + PutM(0);
//			v[1] = (INC(0)) + s + PutM(1);
//			v[2] = (INC(1)) + s + PutM(0);
//			v[3] = s + PutM(1);
//
//			v[4] = s + PutM(2);
//			v[5] = (INC(0)) + s + PutM(2);
//			v[6] = (INC(1) | INC(0)) + s + PutM(2);
//			v[7] = (INC(1)) + s + PutM(2);
//
//			v[8] = (INC(2)) + s + PutM(0);
//			v[9] = (INC(2) | INC(0)) + s + PutM(1);
//			v[10] = (INC(2) | INC(1)) + s + PutM(0);
//			v[11] = (INC(2)) + s + PutM(1);
//
//		}
//		return 12;
//	}
//
//	inline int GetAdjacentCells(Int2Type<VERTEX>, Int2Type<FACE>, index_type s, index_type *v) const
//	{
//		/**
//		 *
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
//		 *
//		 */
//
//		if (v != nullptr)
//		{
//			v[0] = s;
//			v[1] = (DES(2)) + s;
//			v[2] = (DES(2) | DES(1)) + s;
//			v[3] = (DES(1)) + s;
//
//			v[4] = s + PutM(1);
//			v[5] = (DES(2)) + s + PutM(1);
//			v[6] = (DES(0) | DES(2)) + s + PutM(1);
//			v[7] = (DES(0)) + s + PutM(1);
//
//			v[8] = s + PutM(2);
//			v[9] = (DES(1)) + s + PutM(2);
//			v[10] = (DES(1) | DES(0)) + s + PutM(2);
//			v[11] = (DES(0)) + s + PutM(2);
//
//		}
//		return 12;
//	}
//
//	inline int GetAdjacentCells(Int2Type<EDGE>, Int2Type<FACE>, index_type s, index_type *v) const
//	{
//
//		/**
//		 *
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
//		 *
//		 */
//
//		if (v != nullptr)
//		{
//			v[0] = RotateDirection < 1 > (s);
//			v[1] = RotateDirection < 2 > (s);
//			v[2] = RotateDirection < 1 > ((DES(GetM(s) + 2)) + s);
//			v[2] = RotateDirection < 2 > ((DES(GetM(s) + 1)) + s);
//		}
//		return 4;
//	}
//
//	inline int GetAdjacentCells(Int2Type<VOLUME>, Int2Type<FACE>, index_type s, index_type *v) const
//	{
//
//		/**
//		 *
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
//		 *
//		 */
//
//		if (v != nullptr)
//		{
//			v[0] = s + PutM(0);
//			v[1] = (INC(1)) + s + PutM(1);
//			v[2] = (INC(0)) + s + PutM(0);
//			v[3] = s + PutM(1);
//
//			v[4] = s + PutM(2);
//			v[5] = (INC(2)) + s + PutM(2);
//
//		}
//		return 6;
//	}
//
//	inline int GetAdjacentCells(Int2Type<VERTEX>, Int2Type<VOLUME>, index_type s, index_type *v) const
//	{
//		/**
//		 *
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
//		 *
//		 */
//
//		if (v != nullptr)
//		{
//			v[0] = GetMasterVertex(s);
//			v[1] = (DES(0)) + s;
//			v[2] = (DES(0) | DES(1)) + s;
//			v[3] = (DES(1)) + s;
//
//			v[4] = (DES(2)) + s;
//			v[5] = (DES(2) | DES(0)) + s;
//			v[6] = (DES(2) | DES(0) | DES(1)) + s;
//			v[7] = (DES(2) | DES(1)) + s;
//
//		}
//		return 8;
//	}
//
//	inline int GetAdjacentCells(Int2Type<EDGE>, Int2Type<VOLUME>, index_type s, index_type *v) const
//	{
//
//		/**
//		 *
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
//		 *
//		 */
//
//		if (v != nullptr)
//		{
//			v[0] = GetMasterVertex(s);
//			v[1] = (DES(GetM(s) + 1)) + s;
//			v[2] = (DES(GetM(s) + 1) | DES(GetM(s) + 2)) + s;
//			v[3] = (DES(GetM(s) + 2)) + s;
//		}
//		return 4;
//	}
//
//	inline int GetAdjacentCells(Int2Type<FACE>, Int2Type<VOLUME>, index_type s, index_type *v) const
//	{
//
//		/**
//		 *
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
//		 *
//		 */
//
//		if (v != nullptr)
//		{
//			v[0] = GetMasterVertex(s);
//			v[1] = (DES(GetM(s))) + s;
//
//		}
//		return 2;
//	}
//
////***************************************************************************************************
////  Traversal
////
////***************************************************************************************************
//
//	template<typename ... Args>
//	void Traversal(Args const &...args) const
//	{
//		ParallelTraversal(std::forward<Args const &>(args)...);
//	}
//
//	template<typename ...Args> void ParallelTraversal(Args const &...args) const;
//
//	template<typename ...Args> void SerialTraversal(Args const &...args) const;
//
//	void _Traversal(unsigned int num_threads, unsigned int thread_id, int IFORM,
//	        std::function<void(index_type)> const &funs) const;
//
//	template<typename Fun, typename TF, typename ... Args> inline
//	void SerialForEach(Fun const &fun, TF const & l, Args const& ... args) const
//	{
//		SerialTraversal(FieldTraits<TF>::IForm, [&]( index_type s)
//		{	fun(get(l,s),get(args,s)...);});
//	}
//
//	template<typename Fun, typename TF, typename ... Args> inline
//	void SerialForEach(Fun const &fun, TF *l, Args const& ... args) const
//	{
//		if (l == nullptr)
//			ERROR << "Access value to an uninitilized container!";
//
//		SerialTraversal(FieldTraits<TF>::IForm, [&]( index_type s)
//		{	fun(get(l,s),get(args,s)...);});
//	}
//
//	template<typename Fun, typename TF, typename ... Args> inline
//	void ParallelForEach(Fun const &fun, TF const & l, Args const& ... args) const
//	{
//		ParallelTraversal(FieldTraits<TF>::IForm, [&]( index_type s)
//		{	fun(get(l,s),get(args,s)...);});
//	}
//
//	template<typename Fun, typename TF, typename ... Args> inline
//	void ParallelForEach(Fun const &fun, TF *l, Args const& ... args) const
//	{
//		if (l == nullptr)
//			ERROR << "Access value to an uninitilized container!";
//
//		ParallelTraversal(FieldTraits<TF>::IForm, [&]( index_type s)
//		{	fun(get(l,s),get(args,s)...);});
//	}
//
//	template<typename Fun, typename TF, typename ... Args> inline
//	void ForEach(Fun const &fun, TF const & l, Args const& ... args) const
//	{
//		ParallelForEach(fun, l, std::forward<Args const &>(args)...);
//	}
//	template<typename Fun, typename TF, typename ... Args> inline
//	void ForEach(Fun const &fun, TF *l, Args const& ... args) const
//	{
//		ParallelForEach(fun, l, std::forward<Args const &>(args)...);
//	}
//
////***************************************************************************************************
////* Container/Field operation
////* Field vs. Mesh
////***************************************************************************************************
//
//	template<typename TL, typename TR> void AssignContainer(int IFORM, TL * lhs, TR const &rhs) const
//	{
//		ParallelTraversal(IFORM, [&]( index_type s)
//		{	get(lhs,s)=get(rhs,s);});
//
//	}
//
//	template<typename T>
//	inline typename std::enable_if<!is_field<T>::value, T>::type get(T const &l, index_type) const
//	{
//		return std::move(l);
//	}
//

}
;

}
// namespace simpla

#endif /* OCTREE_FOREST_H_ */
