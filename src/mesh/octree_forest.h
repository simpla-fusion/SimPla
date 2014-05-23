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
#include <iterator>
#include "../fetl/ntuple.h"
#include "../fetl/primitives.h"
#include "../utilities/type_utilites.h"
#include "../utilities/pretty_stream.h"

namespace simpla
{

struct OcForest
{

	typedef OcForest this_type;
	static constexpr int MAX_NUM_NEIGHBOUR_ELEMENT = 12;
	static constexpr int MAX_NUM_VERTEX_PER_CEL = 8;
	static constexpr int NDIMS = 3;

	typedef unsigned long size_type;

	typedef size_type compact_index_type;

	struct iterator;

	struct Range;

	typedef nTuple<NDIMS, Real> coordinates_type;

	typedef std::map<iterator, nTuple<3, coordinates_type>> surface_type;

	//!< signed long is 63bit, unsigned long is 64 bit, add a sign bit
	static constexpr unsigned int FULL_DIGITS = std::numeric_limits<compact_index_type>::digits;

	static constexpr unsigned int D_FP_POS = 4; //!< default floating-point position

	static constexpr unsigned int INDEX_DIGITS = (FULL_DIGITS - CountBits<D_FP_POS>::n) / 3;

	static constexpr size_type INDEX_MASK = (1UL << INDEX_DIGITS) - 1;
	static constexpr size_type TREE_ROOT_MASK = ((1UL << (INDEX_DIGITS - D_FP_POS)) - 1) << D_FP_POS;
	static constexpr size_type ROOT_MASK = TREE_ROOT_MASK | (TREE_ROOT_MASK << INDEX_DIGITS)
	        | (TREE_ROOT_MASK << (INDEX_DIGITS * 2));

	static constexpr size_type INDEX_ZERO = ((1UL << (INDEX_DIGITS - D_FP_POS - 1)) - 1) << D_FP_POS;

	//***************************************************************************************************

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

//	nTuple<NDIMS, size_type> global_end_ = { 1, 1, 1 };

	unsigned long clock_ = 0;

	static compact_index_type Compact(nTuple<NDIMS, size_type> const & idx)
	{
		return

		(((idx[0]) & INDEX_MASK) << (INDEX_DIGITS * 2)) |

		(((idx[1]) & INDEX_MASK) << (INDEX_DIGITS)) |

		(((idx[2]) & INDEX_MASK));
	}
	static nTuple<NDIMS, size_type> Decompact(compact_index_type s)
	{
		return nTuple<NDIMS, size_type>( {

		((s >> (INDEX_DIGITS * 2)) & INDEX_MASK),

		((s >> (INDEX_DIGITS)) & INDEX_MASK),

		(s & INDEX_MASK)

		});
	}

	//***************************************************************************************************

	OcForest()
	{
	}

	template<typename TDict>
	OcForest(TDict const & dict)
	{
	}

	~OcForest()
	{
	}

	this_type & operator=(const this_type&) = delete;
	OcForest(const this_type&) = delete;

	void swap(OcForest & rhs)
	{
		//FIXME NOT COMPLETE!!
	}

	template<typename TDict, typename ...Others>
	void Load(TDict const & dict, Others const& ...)
	{
		if (dict["Dimensions"])
		{
			LOGGER << "Load OcForest ";
			SetDimensions(dict["Dimensions"].template as<nTuple<3, size_type>>());

		}

	}

	std::string Save(std::string const &path) const
	{
		std::stringstream os;

		os << "\tDimensions =  " << GetDimensions();

		return os.str();
	}

	void NextTimeStep()
	{
		++clock_;
	}
	unsigned long GetClock() const
	{
		return clock_;
	}

	//***************************************************************************************************
	// Local Data Set

	nTuple<NDIMS, size_type> global_start_, global_count_;

	nTuple<NDIMS, size_type> local_outer_start_, local_outer_count_;

	nTuple<NDIMS, size_type> local_inner_start_, local_inner_count_;

	nTuple<NDIMS, size_type> hash_stride_ = { 0, 0, 0 };

	//
	//   |----------------|----------------|---------------|--------------|------------|
	//   ^                ^                ^               ^              ^            ^
	//   |                |                |               |              |            |
	//global          local_outer      local_inner    local_inner    local_outer     global
	// _start          _start          _start           _end           _end          _end
	//

	enum
	{
		FAST_FIRST, SLOW_FIRST
	};

	int array_order_ = SLOW_FIRST;

	template<typename TI>
	void SetDimensions(TI const &d)
	{
		for (int i = 0; i < NDIMS; ++i)
		{
			size_type length = d[i] > 0 ? d[i] : 1;

			ASSERT(length<INDEX_ZERO );

			global_start_[i] = ((INDEX_ZERO >> D_FP_POS) - length / 2);
			global_count_[i] = length;

		}

		local_outer_start_ = global_start_;
		local_outer_count_ = global_count_;

		local_inner_start_ = global_start_;
		local_inner_count_ = global_count_;
		UpdateHash();
	}

	template<typename ... Args>
	void Decompose(Args const & ... args)
	{
		Range range(local_inner_start_, local_inner_count_, 0UL);

		auto res = range.Split(std::forward<Args const &>(args)...);

		local_outer_start_ = res.first.start_;
		local_outer_count_ = res.first.count_;

		local_inner_start_ = res.second.start_;
		local_inner_count_ = res.second.count_;
		UpdateHash();
	}

	void UpdateHash()
	{
		if (array_order_ == SLOW_FIRST)
		{
			hash_stride_[2] = 1;
			hash_stride_[1] = (local_outer_count_[2]);
			hash_stride_[0] = ((local_outer_count_[1])) * hash_stride_[1];
		}
		else
		{
			hash_stride_[0] = 1;
			hash_stride_[1] = (local_outer_count_[0]);
			hash_stride_[2] = ((local_outer_count_[1])) * hash_stride_[1];
		}
	}

	inline size_type Hash(iterator s) const
	{
		auto d = Decompact((s.self_ & ROOT_MASK) >> D_FP_POS);

		size_type res =

		((d[0] - local_outer_start_[0])) * hash_stride_[0] +

		((d[1] - local_outer_start_[1])) * hash_stride_[1] +

		((d[2] - local_outer_start_[2])) * hash_stride_[2];

		switch (s.NodeId())
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

	nTuple<NDIMS, size_type> const& GetDimensions() const
	{
		return global_count_;
	}

	nTuple<NDIMS, size_type> const& GetLocalDimensions() const
	{
		return local_outer_count_;
	}

	nTuple<NDIMS, Real> GetGlobalExtents() const
	{

		nTuple<NDIMS, Real> res;
		for (int i = 0; i < NDIMS; ++i)
		{
			res[i] = static_cast<Real>(global_count_[i]);
		}
		return res;
	}

	size_type GetNumOfElements(int IFORM = VERTEX) const
	{
		return global_count_[0] * global_count_[1] * global_count_[2] * ((IFORM == VERTEX || IFORM == VOLUME) ? 1 : 3);
	}
	size_type GetLocalNumOfElements(int IFORM = VERTEX) const
	{
		return local_outer_count_[0] * local_outer_count_[1] * local_outer_count_[2]
		        * ((IFORM == VERTEX || IFORM == VOLUME) ? 1 : 3);
	}
	int GetDataSetShape(int IFORM, size_type * global_dims = nullptr, size_type * global_start = nullptr,
	        size_type * local_dims = nullptr, size_type * local_start = nullptr, size_type * local_count = nullptr,
	        size_type * local_stride = nullptr, size_type * local_block = nullptr) const
	{
		int rank = 0;

		for (int i = 0; i < NDIMS; ++i)
		{
			if ((global_count_[i]) > (global_start_[i]))
			{
				if (global_dims != nullptr)
					global_dims[rank] = (global_count_[i]);

				if (global_start != nullptr)
					global_start[rank] = (local_outer_start_[i] - global_start_[i]);

				if (local_dims != nullptr)
					local_dims[rank] = (local_outer_count_[i]);

				if (local_start != nullptr)
					local_start[rank] = (local_inner_start_[i] - local_outer_start_[i]);

				if (local_count != nullptr)
					local_count[rank] = (local_inner_count_[i]);

				++rank;
			}

		}
		if (IFORM == EDGE || IFORM == FACE)
		{
			if (global_dims != nullptr)
				global_dims[rank] = 3;

			if (global_start != nullptr)
				global_start[rank] = 0;

			if (local_dims != nullptr)
				local_dims[rank] = 3;

			if (local_start != nullptr)
				local_start[rank] = 0;

			if (local_count != nullptr)
				local_count[rank] = 3;

			++rank;
		}
		return rank;
	}

	Range GetRange(int IFORM = VERTEX) const
	{
		compact_index_type shift = 0UL;

		if (IFORM == EDGE)
		{
			shift = (_DI >> 1);

		}
		else if (IFORM == FACE)
		{
			shift = ((_DJ | _DK) >> 1);

		}
		else if (IFORM == VOLUME)
		{
			shift = ((_DI | _DJ | _DK) >> 1);

		}

		return Range(global_start_, global_count_, shift);
	}

//***************************************************************************************************

	inline coordinates_type GetCoordinates(iterator const& s) const
	{
		auto d = Decompact(s.self_);

		return coordinates_type( {

		static_cast<Real>(d[0]),

		static_cast<Real>(d[1]),

		static_cast<Real>(d[2])

		});

	}

	coordinates_type CoordinatesLocalToGlobal(iterator const& s, coordinates_type r) const
	{
		Real a = static_cast<double>(1UL << (D_FP_POS - s.HeightOfTree()));
		auto d = Decompact(s.self_);

		return coordinates_type( {

		static_cast<Real>(d[0]) + r[0] * a,

		static_cast<Real>(d[1]) + r[1] * a,

		static_cast<Real>(d[2]) + r[2] * a

		});
	}

//	inline iterator CoordinatesGlobalToLocalDual(coordinates_type *px, compact_index_type shift = 0UL) const
//	{
//		return CoordinatesGlobalToLocal(px, shift, 0.5);
//	}
//	inline iterator CoordinatesGlobalToLocal(coordinates_type *px, compact_index_type shift = 0UL,
//	        double round = 0.0) const
//	{
//		iterator res();
//
//		auto & x = *px;
//
//		compact_index_type h = 0;
//
//		nTuple<NDIMS, long> idx;
//
//		Real w = static_cast<Real>(1UL << h);
//
//		compact_index_type m = (~((1UL << (D_FP_POS - h)) - 1));
//
//		idx[0] = static_cast<long>(std::floor(round + x[0] + static_cast<double>(shift[0]))) & m;
//
//		x[0] = ((x[0] - idx[0]) * w);
//
//		idx[1] = static_cast<long>(std::floor(round + x[1] + static_cast<double>(shift[1]))) & m;
//
//		x[1] = ((x[1] - idx[1]) * w);
//
//		idx[2] = static_cast<long>(std::floor(round + x[2] + static_cast<double>(shift[2]))) & m;
//
//		x[2] = ((x[2] - idx[2]) * w);
//
////				iterator(
////		        { ((((h << (INDEX_DIGITS * 3)) | (idx[0] << (INDEX_DIGITS * 2)) | (idx[1] << (INDEX_DIGITS)) | (idx[2]))
////		                | shift)) })
////
////		;
//
//	}

	static Real Volume(iterator s)
	{
		static constexpr double volume_[8][D_FP_POS] = {

		1, 1, 1, 1, // 000

		        1, 1.0 / 2, 1.0 / 4, 1.0 / 8, // 001

		        1, 1.0 / 2, 1.0 / 4, 1.0 / 8, // 010

		        1, 1.0 / 4, 1.0 / 16, 1.0 / 64, // 011

		        1, 1.0 / 2, 1.0 / 4, 1.0 / 8, // 100

		        1, 1.0 / 4, 1.0 / 16, 1.0 / 64, // 101

		        1, 1.0 / 4, 1.0 / 16, 1.0 / 64, // 110

		        1, 1.0 / 8, 1.0 / 32, 1.0 / 128 // 111

		};

		return volume_[s.NodeId()][s.HeightOfTree()];
	}

	static Real InvVolume(iterator s)
	{
		static constexpr double inv_volume_[8][D_FP_POS] = {

		1, 1, 1, 1, // 000

		        1, 2, 4, 8, // 001

		        1, 2, 4, 8, // 010

		        1, 4, 16, 64, // 011

		        1, 2, 4, 8, // 100

		        1, 4, 16, 64, // 101

		        1, 4, 16, 64, // 110

		        1, 8, 32, 128 // 111

		        };

		return inv_volume_[s.NodeId()][s.HeightOfTree()];
	}

	static Real InvDualVolume(iterator s)
	{
		return InvVolume(s.Dual());
	}
	static Real DualVolume(iterator s)
	{
		return Volume(s.Dual());
	}

	template<int I>
	inline int GetAdjacentCells(Int2Type<I>, Int2Type<I>, iterator s, iterator *v) const
	{
		v[0] = s;
		return 1;
	}

	inline int GetAdjacentCells(Int2Type<EDGE>, Int2Type<VERTEX>, iterator s, iterator *v) const
	{
		v[0] = s + s.DeltaIndex();
		v[1] = s - s.DeltaIndex();
		return 2;
	}

	inline int GetAdjacentCells(Int2Type<FACE>, Int2Type<VERTEX>, iterator s, iterator *v) const
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

		auto di = s.Dual().Roate().DeltaIndex();
		auto dj = s.Dual().InverseRoate().DeltaIndex();

		v[0] = s - di - dj;
		v[1] = s - di - dj;
		v[2] = s + di + dj;
		v[3] = s + di + dj;

		return 4;
	}

	inline int GetAdjacentCells(Int2Type<VOLUME>, Int2Type<VERTEX>, iterator const &s, iterator *v) const
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
		auto di = _DI >> (s.HeightOfTree() + 1);
		auto dj = _DJ >> (s.HeightOfTree() + 1);
		auto dk = _DK >> (s.HeightOfTree() + 1);

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

	inline int GetAdjacentCells(Int2Type<VERTEX>, Int2Type<EDGE>, iterator s, iterator *v) const
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

		auto di = _DI >> (s.HeightOfTree() + 1);
		auto dj = _DJ >> (s.HeightOfTree() + 1);
		auto dk = _DK >> (s.HeightOfTree() + 1);

		v[0] = s + di;
		v[1] = s - di;

		v[2] = s + dj;
		v[3] = s - dj;

		v[4] = s + dk;
		v[5] = s - dk;

		return 6;
	}

	inline int GetAdjacentCells(Int2Type<FACE>, Int2Type<EDGE>, iterator s, iterator *v) const
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
		auto d1 = s.Dual().Roate().DeltaIndex();
		auto d2 = s.Dual().InverseRoate().DeltaIndex();
		v[0] = s - d1;
		v[1] = s + d1;
		v[2] = s - d2;
		v[3] = s + d2;

		return 4;
	}

	inline int GetAdjacentCells(Int2Type<VOLUME>, Int2Type<EDGE>, iterator s, iterator *v) const
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
		auto di = _DI >> (s.HeightOfTree() + 1);
		auto dj = _DJ >> (s.HeightOfTree() + 1);
		auto dk = _DK >> (s.HeightOfTree() + 1);

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

	inline int GetAdjacentCells(Int2Type<VERTEX>, Int2Type<FACE>, iterator s, iterator *v) const
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
		auto di = _DI >> (s.HeightOfTree() + 1);
		auto dj = _DJ >> (s.HeightOfTree() + 1);
		auto dk = _DK >> (s.HeightOfTree() + 1);

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

	inline int GetAdjacentCells(Int2Type<EDGE>, Int2Type<FACE>, iterator s, iterator *v) const
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

		auto d1 = s.Roate().DeltaIndex();
		auto d2 = s.InverseRoate().DeltaIndex();

		v[0] = s - d1;
		v[1] = s + d1;
		v[2] = s - d2;
		v[3] = s + d2;

		return 4;
	}

	inline int GetAdjacentCells(Int2Type<VOLUME>, Int2Type<FACE>, iterator s, iterator *v) const
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

		auto di = _DI >> (s.HeightOfTree() + 1);
		auto dj = _DJ >> (s.HeightOfTree() + 1);
		auto dk = _DK >> (s.HeightOfTree() + 1);

		v[0] = s - di;
		v[1] = s + di;

		v[2] = s - di;
		v[3] = s + dj;

		v[4] = s - dk;
		v[5] = s + dk;

		return 6;
	}

	inline int GetAdjacentCells(Int2Type<VERTEX>, Int2Type<VOLUME>, iterator s, iterator *v) const
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

		auto di = _DI >> (s.HeightOfTree() + 1);
		auto dj = _DJ >> (s.HeightOfTree() + 1);
		auto dk = _DK >> (s.HeightOfTree() + 1);

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

	inline int GetAdjacentCells(Int2Type<EDGE>, Int2Type<VOLUME>, iterator s, iterator *v) const
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

		auto d1 = s.Roate().DeltaIndex();
		auto d2 = s.InverseRoate().DeltaIndex();

		v[0] = s - d1 - d2;
		v[1] = s + d1 - d2;
		v[2] = s - d1 + d2;
		v[3] = s + d1 + d2;
		return 4;
	}

	inline int GetAdjacentCells(Int2Type<FACE>, Int2Type<VOLUME>, iterator s, iterator *v) const
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

		auto d = s.Dual().DeltaIndex();
		v[0] = s + d;
		v[1] = s - d;

		return 2;
	}

	struct iterator
	{
/// One of the @link iterator_tags tag types@endlink.
		typedef std::input_iterator_tag iterator_category;

/// The type "pointed to" by the iterator.
		typedef compact_index_type value_type;

/// Distance between iterators is represented as this type.
		typedef typename simpla::OcForest::iterator difference_type;

/// This type represents a pointer-to-value_type.
		typedef value_type* pointer;

/// This type represents a reference-to-value_type.
		typedef value_type& reference;

		compact_index_type self_;

		compact_index_type start_, end_;

		iterator(compact_index_type s = 0, compact_index_type b = 0, compact_index_type e = 0)
				: self_(s), start_(b), end_(e)
		{
		}

		~iterator()
		{
		}

		bool operator==(iterator const & rhs) const
		{
			return self_ == rhs.self_;
		}

		bool operator!=(iterator const & rhs) const
		{
			return !(this->operator==(rhs));
		}

		iterator const & operator*() const
		{
			return *this;
		}

		iterator const* operator ->() const
		{
			return this;
		}

		void NextCell()
		{
			auto D = (1UL << (D_FP_POS - HeightOfTree()));

			self_ += D;

			if ((self_ & _MRK) >= (end_ & _MRK))
			{
				self_ &= ~_MRK;
				self_ |= start_ & _MRK;
				self_ += D << (INDEX_DIGITS);
			}
			if ((self_ & _MRJ) >= (end_ & _MRJ))
			{
				self_ &= ~_MRJ;
				self_ |= start_ & _MRJ;
				self_ += D << (INDEX_DIGITS * 2);
			}

		}

		void PreviousCell()
		{
			auto D = (1UL << (D_FP_POS - HeightOfTree()));

			self_ -= D;

			if ((self_ & _MRK) < (start_ & _MRK))
			{
				self_ &= ~_MRK;
				self_ |= (end_ - D) & _MRK;
				self_ -= D << (INDEX_DIGITS);
			}
			if ((self_ & _MRJ) < (end_ & _MRJ))
			{
				self_ &= ~_MRJ;
				self_ |= (end_ - (D << INDEX_DIGITS)) & _MRK;
				self_ -= D << (INDEX_DIGITS * 2);
			}

		}

		iterator & operator ++()
		{
			auto n = NodeId();

			if (n == 0 || n == 4 || n == 3 || n == 7)
			{
				NextCell();
			}

			self_ = Roate(self_);

			return *this;
		}
		iterator operator ++(int)
		{
			iterator res(*this);
			++res;
			return std::move(res);
		}

		iterator & operator --()
		{

			auto n = NodeId();

			if (n == 0 || n == 1 || n == 6 || n == 7)
			{
				PreviousCell();
			}

			self_ = InverseRoate(self_);

			return *this;
		}

		iterator operator --(int)
		{
			iterator res(*this);
			--res;
			return std::move(res);
		}

#define DEF_OP(_OP_)                                                                       \
				inline iterator & operator _OP_##=(compact_index_type r)                           \
				{                                                                                  \
					self_ =  ( (*this) _OP_ r).self_;                                                                \
					return *this ;                                                                  \
				}                                                                                  \
				inline iterator &operator _OP_##=(iterator r)                                   \
				{                                                                                  \
					self_ = ( (*this) _OP_ r).self_;                                                                  \
					return *this;                                                                  \
				}                                                                                  \
		                                                                                           \
				inline iterator  operator _OP_(compact_index_type const &r) const                 \
				{   iterator res(*this);                                                                               \
				   res.self_=(( ( ((self_ _OP_ (r & _MI)) & _MI) |                              \
				                     ((self_ _OP_ (r & _MJ)) & _MJ) |                               \
				                     ((self_ _OP_ (r & _MK)) & _MK)                                 \
				                        )& (NO_HEAD_FLAG) ));   \
			      return res;                                              \
				}                                                                                  \
		                                                                                           \
				inline iterator operator _OP_(iterator r) const                                \
				{                                                                                  \
					return std::move(this->operator _OP_(r.self_));                                               \
				}                                                                                  \

		DEF_OP(+)
		DEF_OP(-)
		DEF_OP(^)
		DEF_OP(&)
		DEF_OP(|)
#undef DEF_OP

		//***************************************************************************************************
		//* Auxiliary functions
		//***************************************************************************************************
		void SetNodeId(unsigned int n)
		{

		}

		iterator Root() const
		{
			iterator res(*this);
			res.self_ = self_ & ROOT_MASK;
			//			compact_index_type m = (1 << (D_FP_POS - HeightOfTree())) - 1;
			//			return iterator( { d & (~((m << INDEX_DIGITS * 2) | (m << (INDEX_DIGITS)) | m)) });
			return res;
		}
		iterator Dual() const
		{
			iterator res(*this);

			res.self_ = (self_ & (~(_DA >> (HeightOfTree() + 1))))
			        | ((~(self_ & (_DA >> (HeightOfTree() + 1)))) & (_DA >> (HeightOfTree() + 1)));

			return res;
		}

		unsigned int HeightOfTree() const
		{
			return self_ >> (INDEX_DIGITS * 3);
		}

		unsigned int NodeId() const
		{
			auto s = (self_ & (_DA >> (HeightOfTree() + 1))) >> (D_FP_POS - HeightOfTree() - 1);

			return ((s >> (INDEX_DIGITS * 2)) | (s >> (INDEX_DIGITS - 1)) | (s << 2UL)) & (7UL);
		}

		/**
		 *  rotate vector direction  mask
		 *  (1/2,0,0) => (0,1/2,0) or   (1/2,1/2,0) => (0,1/2,1/2)
		 * @param s
		 * @return
		 */
		iterator Roate() const
		{
			iterator res(*this);

			res.self_ = Roate(self_);

			return res;
		}

		compact_index_type Roate(compact_index_type r) const
		{

			compact_index_type res;

			res = r & (~(_DA >> (HeightOfTree() + 1)));

			res |= ((r & ((_DI | _DJ) >> (HeightOfTree() + 1))) >> INDEX_DIGITS) |

			((r & (_DK >> (HeightOfTree() + 1))) << (INDEX_DIGITS * 2))

			;
			return res;

		}

		/**
		 *  rotate vector direction  mask
		 *  (1/2,0,0) => (0,0,1/2) or   (1/2,1/2,0) => (1/2,0,1/2)
		 * @param s
		 * @return
		 */
		compact_index_type InverseRoate(compact_index_type) const
		{
			compact_index_type res;

			res = self_ & ~(_DA >> (HeightOfTree() + 1));

			res |= ((self_ & (_DI >> (HeightOfTree() + 1))) >> (INDEX_DIGITS * 2)) |

			((self_ & (_DJ >> (HeightOfTree() + 1))) << INDEX_DIGITS) |

			((self_ & (_DK >> (HeightOfTree() + 1))) << INDEX_DIGITS);

			return res;
		}

		iterator InverseRoate() const
		{
			iterator res(*this);

			res.self_ = InverseRoate(self_);

			return res;
		}

		//! get the direction of vector(edge) 0=>x 1=>y 2=>z
		compact_index_type DirectionOfVector() const
		{
			compact_index_type s = (self_ & (_DA >> (HeightOfTree() + 1))) >> (D_FP_POS - HeightOfTree() - 1);

			return ((s >> (INDEX_DIGITS * 2)) | (s >> (INDEX_DIGITS - 1)) | (s << 2UL)) & (7UL);
		}

		compact_index_type DeltaIndex() const
		{
			return (self_ & (_DA >> (HeightOfTree() + 1)));
		}

		compact_index_type DeltaIndex(unsigned int i) const
		{
			return (1UL << (INDEX_DIGITS * (NDIMS - i - 1) + D_FP_POS - HeightOfTree() - 1));
		}

		/**
		 * Get component number or vector direction
		 * @param s
		 * @return
		 */
		size_type ComponentNum() const
		{
			size_type res = 0;
			switch (NodeId())
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

		size_type IForm() const
		{
			size_type res = 0;
			switch (NodeId())
			{
			case 0:
				res = VERTEX;
				break;
			case 1:
			case 2:
			case 4:
				res = EDGE;
				break;

			case 3:
			case 5:
			case 6:
				res = FACE;
				break;

			case 7:
				res = VOLUME;
			}
			return res;
		}

	}; // class iterator

	struct Range
	{
	public:
		typedef typename OcForest::iterator iterator;

		nTuple<NDIMS, size_type> start_ = { 0, 0, 0 }, count_ = { 0, 0, 0 };
		compact_index_type shift_ = 0UL;

		Range()
		{

		}
		Range(nTuple<NDIMS, size_type> const & start, nTuple<NDIMS, size_type> const& count,
		        compact_index_type node_shift = 0UL)
				: start_(start), count_(count), shift_(node_shift)
		{
		}

		~Range()
		{
		}

		iterator begin() const
		{
			return iterator((Compact(start_) << D_FP_POS) | shift_, ((Compact(start_) << D_FP_POS) | shift_),
			        ((Compact(start_ + count_) << D_FP_POS) | shift_));
		}
		iterator end() const
		{
			iterator res(shift_, shift_, shift_);

			if (count_[0] * count_[1] * count_[2] > 0)
			{
				res = iterator(

				(Compact(start_ + count_ - 1) << D_FP_POS) | shift_,

				((Compact(start_) << D_FP_POS) | shift_),

				((Compact(start_ + count_) << D_FP_POS) | shift_)

				);
				res.NextCell();
			}
			return res;
		}
		nTuple<NDIMS, size_type> const& Extents() const
		{
			return count_;
		}
		size_type Size() const
		{
			return size();
		}
		size_type size() const
		{
			size_type n = 1;

			for (int i = 0; i < NDIMS; ++i)
			{
				n *= count_[i];
			}
			return n;
		}
		Range Split(unsigned int total, unsigned int sub, unsigned int gw = 0) const
		{
			Range res;
			nTuple<NDIMS, size_type> num_process;
			nTuple<NDIMS, size_type> process_num;
			nTuple<NDIMS, size_type> ghost_width;

			auto extents = Extents();

			bool flag = false;
			for (int i = 0; i < NDIMS; ++i)
			{
				ghost_width[i] = gw;
				if (!flag && (extents[i] > total))
				{
					num_process[i] = total;
					process_num[i] = sub;
					flag = true;
				}
				else
				{
					num_process[i] = 1;
					process_num[i] = 0;
				}
			}
			if (!flag)
			{
				if (sub == 0)
				{
					WARNING << "I'm the master!";
					res = *this;
				}
				else
				{
					WARNING << "Range is too small to split!  ";
				}
			}
			else
			{
				res = Split(num_process, process_num, ghost_width).first;
			}

			return res;

		}

		std::pair<Range, Range> Split(nTuple<NDIMS, size_type> const & num_process,
		        nTuple<NDIMS, size_type> const & process_num, nTuple<NDIMS, size_type> const & ghost_width) const
		{

			nTuple<NDIMS, size_type>

			inner_start = start_,

			inner_count = count_,

			outer_start, outer_count;

			for (int i = 0; i < NDIMS; ++i)
			{

				if (2 * ghost_width[i] * num_process[i] > inner_count[i])
				{
					ERROR << "Mesh is too small to decompose! dims[" << i << "]=" << inner_count[i]

					<< " process[" << i << "]=" << num_process[i] << " ghost_width=" << ghost_width[i];
				}
				else
				{

					auto start = (inner_count[i] * process_num[i]) / num_process[i];

					auto end = (inner_count[i] * (process_num[i] + 1)) / num_process[i];

					inner_start[i] += start;
					inner_count[i] = end - start;

					outer_start[i] = inner_start[i];
					outer_count[i] = inner_count[i];

					if (process_num[i] > 0)
					{
						outer_start[i] -= ghost_width[i];
						outer_count[i] += ghost_width[i];

					}
					if (process_num[i] < num_process[i] - 1)
					{
						outer_count[i] += ghost_width[i];

					};

				}
			}

			return std::make_pair(Range(outer_start, outer_count, shift_), Range(inner_start, inner_count, shift_));
		}
	};
	// class Range
};
// class OcForest
}
// namespace simpla

#endif /* OCTREE_FOREST_H_ */
