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

		os << "\tDimensions =  " << GetGlobalDimensions();

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

	nTuple<NDIMS, size_type> global_start_, global_end_;

	nTuple<NDIMS, size_type> local_outer_start_, local_outer_end_;

	nTuple<NDIMS, size_type> local_inner_start_, local_inner_end_;

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

			global_start_[i] = ((INDEX_ZERO >> D_FP_POS) - length / 2) << D_FP_POS;
			global_end_[i] = (global_start_[i] + ((length) << D_FP_POS));

		}

		local_outer_start_ = global_start_;
		local_outer_end_ = global_end_;

		local_inner_start_ = global_start_;
		local_inner_end_ = global_end_;
	}

	nTuple<NDIMS, size_type> GetDimensions() const
	{
		return std::move(GetGlobalDimensions());
	}

	nTuple<NDIMS, size_type> GetGlobalDimensions() const
	{
		nTuple<NDIMS, size_type> count = (global_end_ - global_start_) >> D_FP_POS;
		return count;
	}

	nTuple<NDIMS, size_type> GetLocalDimensions() const
	{
		nTuple<NDIMS, size_type> count = (local_outer_end_ - local_outer_start_) >> D_FP_POS;
		return count;
	}

	nTuple<NDIMS, Real> GetGlobalExtents() const
	{
		auto dims = GetGlobalDimensions();

		return nTuple<NDIMS, Real>( { static_cast<Real>(dims[0]),

		static_cast<Real>(dims[1]),

		static_cast<Real>(dims[2])

		});
	}

	inline size_type Hash(iterator s) const
	{
		auto d = Decompact(s.self_);

		size_type res =

		((d[0] - local_outer_start_[0]) >> D_FP_POS) * hash_stride_[0] +

		((d[1] - local_outer_start_[1]) >> D_FP_POS) * hash_stride_[1] +

		((d[2] - local_outer_start_[2]) >> D_FP_POS) * hash_stride_[2];

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
	void Decompose(nTuple<NDIMS, size_type> const & num_process, nTuple<NDIMS, size_type> const & process_num,
	        nTuple<NDIMS, size_type> const & ghost_width)
	{

		if (array_order_ == SLOW_FIRST)
		{
			hash_stride_[2] = 1;
			hash_stride_[1] = (local_outer_end_[2] - local_outer_start_[2]) >> D_FP_POS;
			hash_stride_[0] = ((local_outer_end_[1] - local_outer_start_[1]) >> D_FP_POS) * hash_stride_[1];
		}
		else
		{
			hash_stride_[0] = 1;
			hash_stride_[1] = (local_outer_end_[0] - local_outer_start_[0]) >> D_FP_POS;
			hash_stride_[2] = ((local_outer_end_[1] - local_outer_start_[1]) >> D_FP_POS) * hash_stride_[1];
		}

	}

	size_type GetNumOfElements(int IFORM = VERTEX) const
	{
		auto dims = GetGlobalDimensions();
		return dims[0] * dims[1] * dims[2] * ((IFORM == VERTEX || IFORM == VOLUME) ? 1 : 3);
	}

	int GetDataSetShape(int IFORM, size_type * global_dims = nullptr, size_type * global_start = nullptr,
	        size_type * local_dims = nullptr, size_type * local_start = nullptr, size_type * local_count = nullptr,
	        size_type * local_stride = nullptr, size_type * local_block = nullptr) const
	{
		int rank = 0;

		for (int i = 0; i < NDIMS; ++i)
		{
			size_type L = (global_end_[i] - global_start_[i]) >> D_FP_POS;
			if (L > 1)
			{
				if (global_dims != nullptr)
					global_dims[rank] = L;

				if (global_start != nullptr)
					global_start[rank] = (local_outer_start_[i] - global_start_[i]) >> D_FP_POS;

				if (local_dims != nullptr)
					local_dims[rank] = (local_outer_end_[i] - local_outer_start_[i]) >> D_FP_POS;

				if (local_start != nullptr)
					local_start[rank] = (local_inner_start_[i] - local_outer_start_[i]) >> D_FP_POS;

				if (local_count != nullptr)
					local_count[rank] = (local_inner_end_[i] - local_inner_start_[i]) >> D_FP_POS;

//				if (local_stride != nullptr)
//					local_stride[rank] = stride_[i];

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

//			if (local_stride != nullptr)
//				local_stride[rank] = 1;

			++rank;
		}
		return rank;
	}

	Range GetRange(int IFORM = VERTEX) const
	{
		compact_index_type b = Compact(global_start_), e = Compact(global_end_);

		if (IFORM == EDGE)
		{
			b |= (_DI >> 1);
			e |= (_DI >> 1);
		}
		else if (IFORM == FACE)
		{
			b |= ((_DJ | _DK) >> 1);
			e |= ((_DJ | _DK) >> 1);
		}
		else if (IFORM == VOLUME)
		{
			b |= ((_DI | _DJ | _DK) >> 1);
			e |= ((_DI | _DJ | _DK) >> 1);
		}

		return Range(b, e);
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
////		idx[0] = static_cast<long>(std::floor(round + x[0] + static_cast<double>(shift[0]))) & m;
////
////		x[0] = ((x[0] - idx[0]) * w);
////
////		idx[1] = static_cast<long>(std::floor(round + x[1] + static_cast<double>(shift[1]))) & m;
////
////		x[1] = ((x[1] - idx[1]) * w);
////
////		idx[2] = static_cast<long>(std::floor(round + x[2] + static_cast<double>(shift[2]))) & m;
////
////		x[2] = ((x[2] - idx[2]) * w);
//
//		return res;
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

		iterator(compact_index_type s, compact_index_type b, compact_index_type e)
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
//
//				if ((self_ & _MRI) >= (end_ & _MRI))
//				{
//					self_ &= ~_MRI;
//					self_ |= end_ & _MRI;
//				}
			//			if (s[0] > end_[0])
			//			{
			//				s.d = -1; // the end
			//			}
		}
		iterator operator ++(int)
		{
			iterator res(*this);
			++res;
			return std::move(res);
		}

		iterator & operator --()
		{

//		//   NEED OPTIMIZE!
//		auto n = self_.NodeId();
//
//		if (n == 0 || n == 4 || n == 3 || n == 7)
//		{
//			auto D = (1UL << (D_FP_POS - self_.HeightOfTree()));
//
//			auto mask = self_ & (~ROOT_MASK);
//
//			self_.Set(2, self_[2] - D);
//
//			if (self_[2] < (start_[2]))
//			{
//				self_.Set(2, end_[2] - D);
//				self_.Set(1, self_[1] - D);
//			}
//			if (self_[1] < (start_[1]))
//			{
//				self_.Set(1, end_[1] - D);
//				self_.Set(0, self_[0] - D);
//			}
//			//			if (s_[0] > end_[0])
//			//			{
//			//				s.d = -1; // the end
//			//			}
//			self_ |= mask;
//		}
//
//		iterator r;
//
//		r.d = self_.d & ~(_DA >> (self_.HeightOfTree() + 1));
//
//		r |= ((self_.d & (_DI >> (self_.HeightOfTree() + 1))) >> (INDEX_DIGITS * 2)) |
//
//		((self_.d & (_DJ >> (self_.HeightOfTree() + 1))) << INDEX_DIGITS) |
//
//		((self_.d & (_DK >> (self_.HeightOfTree() + 1))) << INDEX_DIGITS)
//
//		;
//
//		self_ = r;

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

			res.self_ = Roate(res.self_);

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
		iterator InverseRoate() const
		{
			iterator res(*this);

			res.self_ = self_ & ~(_DA >> (HeightOfTree() + 1));

			res.self_ |= ((self_ & (_DI >> (HeightOfTree() + 1))) >> (INDEX_DIGITS * 2)) |

			((self_ & (_DJ >> (HeightOfTree() + 1))) << INDEX_DIGITS) |

			((self_ & (_DK >> (HeightOfTree() + 1))) << INDEX_DIGITS);

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

		compact_index_type first, second;

		Range()
				: first(0UL), second(0UL)
		{

		}
		Range(nTuple<NDIMS, size_type> const & start, nTuple<NDIMS, size_type> const& count,
		        compact_index_type node_shift = 0UL)
				: first((Compact(start) << D_FP_POS) | node_shift), second(
				        (Compact(start + count) << D_FP_POS) | node_shift)
		{
		}

		Range(compact_index_type b, compact_index_type e)
				: first((b)), second(e)
		{

		}

		~Range()
		{
		}

		iterator begin() const
		{
			return iterator((first), (first), (second));
		}
		iterator end() const
		{
			iterator res(second, (first), (second));
			if (first != second)
			{
				res.self_ -= _DA;
				res.NextCell();
			}
			return res;
		}
		nTuple<NDIMS, size_type> Extents() const
		{
			nTuple<NDIMS, size_type> res;
			res = Decompact(second >> D_FP_POS) - Decompact(first >> D_FP_POS);
			return res;
		}
		size_type Size() const
		{
			return size();
		}
		size_type size() const
		{
			size_type count = 1;
			auto dims = Extents();
			for (int i = 0; i < NDIMS; ++i)
			{
				count *= dims[i];
			}
			return count;
		}
		Range Split(unsigned int total, unsigned int sub) const
		{
			Range res;
			nTuple<NDIMS, size_type> num_process;
			nTuple<NDIMS, size_type> process_num;
			nTuple<NDIMS, size_type> gw;

			auto extents = Extents();

			bool flag = false;
			bool is_master = true;
			for (int i = 0; i < NDIMS; ++i)
			{
				is_master = is_master && (process_num[i] == 0);
				gw[i] = 0;
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
				WARNING << "Range is too small to split!  ";
				if (is_master)
				{
					res = Range(first, second);
				}
			}
			else
			{
				res = Split(num_process, process_num, gw).first;
			}

			return res;

		}

		std::pair<Range, Range> Split(nTuple<NDIMS, size_type> const & num_process,
		        nTuple<NDIMS, size_type> const & process_num, nTuple<NDIMS, size_type> const & ghost_width) const
		{

			compact_index_type inner_start = first, inner_end = first, outer_start = first, outer_end = first;

			auto count = Extents();

			for (int i = 0; i < NDIMS; ++i)
			{

				if (2 * ghost_width[i] * num_process[i] > count[i])
				{
					ERROR << "Mesh is too small to decompose! dims[" << i << "]=" << count[i]

					<< " process[" << i << "]=" << num_process[i] << " ghost_width=" << ghost_width[i];
				}
				else
				{

					auto start = (count[i] * process_num[i]) / num_process[i];

					auto end = (count[i] * (process_num[i] + 1)) / num_process[i];

					inner_start += start << (INDEX_DIGITS * (NDIMS - i - 1) + D_FP_POS);
					inner_end += end << (INDEX_DIGITS * (NDIMS - i - 1) + D_FP_POS);

					if (process_num[i] > 0)
					{
						start -= ghost_width[i] << D_FP_POS;

					}
					if (process_num[i] < num_process[i] - 1)
					{
						end += ghost_width[i] << D_FP_POS;
					}

					outer_start += start << (INDEX_DIGITS * (NDIMS - i - 1) + D_FP_POS);
					outer_end += end << (INDEX_DIGITS * (NDIMS - i - 1) + D_FP_POS);
				}
			}

			return std::make_pair(Range(outer_start, outer_end), Range(inner_start, inner_end));
		}
	};
	// class Range
};
// class OcForest
}
// namespace simpla

#endif /* OCTREE_FOREST_H_ */
