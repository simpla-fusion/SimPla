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
#include "../fetl/ntuple_ops.h"
#include "../fetl/primitives.h"
#include "../utilities/type_utilites.h"

namespace simpla
{

template<typename TO, typename TI>
TO Convert(TI const &x)
{
	union
	{
		TI in;
		TO out;
	};

	in = x;

	return out;
}

template<unsigned int DEFAULT_TREE_HEIGHT>
class OcForest
{

public:
	typedef OcForest this_type;
	static constexpr int MAX_NUM_VERTEX_PER_CEL = 8;
	static constexpr int NUM_OF_DIMS = 3;

	OcForest()
			: MASK_A(Convert<compact_index_type>(index_type( { 0, 1UL, 1UL, 1UL }))),

			MASK_I(Convert<compact_index_type>(index_type( { 0, 1UL, 0, 0 }))),

			MASK_J(Convert<compact_index_type>(index_type( { 0, 0, 1UL, 0 }))),

			MASK_K(Convert<compact_index_type>(index_type( { 0, 0, 0, 1UL })))
	{
	}

	~OcForest()
	{
	}

	this_type & operator=(const this_type&) = delete;

	typedef unsigned long size_type;

	typedef unsigned long compact_index_type;

	static constexpr int DIGITS_FULL = std::numeric_limits<unsigned long>::digits;

	static constexpr int TREE_DEPTH = DEFAULT_TREE_HEIGHT;

	static constexpr int DIGITS_INDEX = (DIGITS_FULL - CountBits<TREE_DEPTH>::n) / 3; //!< signed long is 63bit, unsigned long is 64 bit, add a sign bit

	static constexpr int DIGITS_HEAD = DIGITS_FULL - DIGITS_INDEX * 3;

	struct index_type
	{
		size_type H :DIGITS_HEAD;
		size_type I :DIGITS_INDEX;
		size_type J :DIGITS_INDEX;
		size_type K :DIGITS_INDEX;
	};

	const compact_index_type MASK_A;
	const compact_index_type MASK_I;
	const compact_index_type MASK_J;
	const compact_index_type MASK_K;

	static compact_index_type _C(index_type const &s)
	{
		return Convert<compact_index_type>(s);
	}

	static constexpr size_type INDEX_MAX = static_cast<size_type>(((1L) << (DIGITS_INDEX)) - 1);

	static constexpr size_type INDEX_MIN = 0;

	static constexpr double dh = 1.0 / static_cast<double>(INDEX_MAX + 1);

	inline size_type HashRootIndex(index_type s, size_type strides[3],
	        const unsigned int forest_depth = DIGITS_INDEX - DEFAULT_TREE_HEIGHT) const
	{

		return (

		(s.I >> (DIGITS_INDEX - forest_depth)) * strides[0] +

		(s.J >> (DIGITS_INDEX - forest_depth)) * strides[1] +

		(s.K >> (DIGITS_INDEX - forest_depth)) * strides[2]

		);

	}
	inline index_type GetIndex(nTuple<3, Real> const & x) const
	{
		index_type res;

		ASSERT(0<=x[0] && x[0]<=1.0);

		ASSERT(0<=x[1] && x[1]<=1.0);

		ASSERT(0<=x[2] && x[2]<=1.0);

		res.I = static_cast<size_type>(std::floor(x[0] * static_cast<double>(INDEX_MAX + 1)));

		res.J = static_cast<size_type>(std::floor(x[1] * static_cast<double>(INDEX_MAX + 1)));

		res.K = static_cast<size_type>(std::floor(x[2] * static_cast<double>(INDEX_MAX + 1)));

		return res;
	}

	inline index_type GetIndex(nTuple<3, Real> const & x, nTuple<3, Real> * r, unsigned int tree_height =
	        DEFAULT_TREE_HEIGHT) const
	{
		index_type res = GetIndex(x);
		size_type m = (~0L) << DEFAULT_TREE_HEIGHT;
		res.I &= m;
		res.J &= m;
		res.K &= m;

		*r = x - GetCoordinates(res);

		return res;
	}

	inline nTuple<3, Real> GetCoordinates(index_type s) const
	{

		return nTuple<3, Real>( {

		static_cast<double>(s.I) * dh,

		static_cast<double>(s.J) * dh,

		static_cast<double>(s.K) * dh

		});

	}

//	/**
//	 * 	Thanks my wife Dr. CHEN Xiang Lan, for her advice on  these bitwise operation
//	 * 	               m            m             m
//	 * 	|--------|------------|--------------|-------------|
//	 * 	               I              J             K
//	 *
//	 * 	 n+m*3=digits(unsigned long)=63
//	 * 	 n=3
//	 * 	 m=20
//	 */
//
//	inline index_type INC(int m, index_type s) const
//	{
//		return s + (1L << ((m % 3) * DIGITS_INDEX));
//	}
//	inline index_type DES(int m, index_type s) const
//	{
//		return s - (1L << ((m % 3) * DIGITS_INDEX));
//	}
//
//	inline size_type HashIndex(index_type s, size_type strides[3]) const
//	{
//		return (s.I * strides[0] + s.J * strides[1] + s.K * strides[2]);
//
//	}
//
//	inline index_type GetRoot(index_type s) const
//	{
//		return s & BIT_MASK_ROOT;
//	}
//	inline index_type GetTree(index_type s) const
//	{
//		return s & BIT_MASK_TREE;
//	}
////***************************************************************************************************
//
	template<int I>
	inline int GetAdjacentCells(Int2Type<I>, Int2Type<I>, index_type s, index_type *v) const
	{
		if (v != nullptr)
			v[0] = s;
		return 1;
	}

	inline int GetAdjacentCells(Int2Type<EDGE>, Int2Type<VERTEX>, index_type s, index_type *v) const
	{
		if (v != nullptr)
		{
			_C(v[0]) = _C(s) & (~MASK_A);
			_C(v[1]) = _C(v[0]) + ((_C(s) & MASK_A) << 1);
		}
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

		if (v != nullptr)
		{
			_C(v[0]) = _C(s) & (~MASK_A);
			_C(v[1]) = _C(v[0]) + ((_C(s) & MASK_I) << 1);
			_C(v[3]) = _C(v[0]) + ((_C(s) & MASK_J) << 1);
			_C(v[2]) = _C(v[0]) + ((_C(s) & MASK_J) << 1);
		}
		return 4;
	}
//
//	inline int GetAdjacentCells(Int2Type<VOLUME>, Int2Type<VERTEX>, index_type s, index_type *v) const
//	{
//		/**
//		 *
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
//		 *
//		 */
//
//		if (v != nullptr)
//		{
//			v[0] = GetMasterVertex(s);
//			v[1] = (INC(0)) + s;
//			v[2] = (INC(1) | INC(1)) + s;
//			v[3] = (INC(1)) + s;
//
//			v[4] = (INC(2)) + s;
//			v[5] = (INC(2) | INC(0)) + s;
//			v[6] = (INC(2) | INC(1) | INC(1)) + s;
//			v[7] = (INC(2) | INC(1)) + s;
//		}
//		return 8;
//	}
//
//	inline int GetAdjacentCells(Int2Type<VERTEX>, Int2Type<EDGE>, index_type s, index_type *v) const
//	{
//		/**
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
//		 *
//		 */
//
//		if (v != nullptr)
//		{
//			v[0] = s + PutM(0);
//			v[1] = s + PutM(1);
//			v[2] = s + PutM(2);
//			v[3] = (DES(1)) + s + PutM(0);
//			v[4] = (DES(2)) + s + PutM(1);
//			v[5] = (DES(2)) + s + PutM(2);
//		}
//		return 6;
//	}
//
//	inline int GetAdjacentCells(Int2Type<FACE>, Int2Type<EDGE>, index_type s, index_type *v) const
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
//		 *
//		 */
//
//		if (v != nullptr)
//		{
//			v[0] = RotateDirection < 1 > (s);
//			v[1] = RotateDirection < 2 > (s);
//			v[2] = RotateDirection < 1 > ((INC(GetM(s) + 1)) + s);
//			v[2] = RotateDirection < 2 > ((INC(GetM(s) + 2)) + s);
//		}
//		return 4;
//	}
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
