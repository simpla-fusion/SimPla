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
#include "../utilities/memory_pool.h"
#include "../parallel/distributed_array.h"

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

	struct range;

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

	static constexpr Real R_INDEX_ZERO = static_cast<Real>(INDEX_ZERO);

	static constexpr Real R_INV_DX = static_cast<Real>(1UL << D_FP_POS);
	static constexpr Real R_DX = 1.0 / R_INV_DX;
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
	static nTuple<NDIMS, size_type> DecompactRoot(compact_index_type s)
	{
		return (Decompact(s) - (((1UL << (INDEX_DIGITS - D_FP_POS - 1)) - 1) << D_FP_POS)) >> D_FP_POS;

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

	template<typename TV> using Container=std::shared_ptr<TV>;

	template<int iform, typename TV> inline std::shared_ptr<TV> MakeContainer() const
	{
		return (MEMPOOL.allocate_shared_ptr < TV > (GetLocalNumOfElements(iform)));
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

	unsigned long clock_ = 0UL;

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

	nTuple<NDIMS, size_type> hash_stride_ =
	{	0, 0, 0};

	enum
	{
		FAST_FIRST, SLOW_FIRST
	};

	int array_order_ = SLOW_FIRST;

	DistributedArray<NDIMS> global_array_;
	//
	//   |----------------|----------------|---------------|--------------|------------|
	//   ^                ^                ^               ^              ^            ^
	//   |                |                |               |              |            |
	//global          local_outer      local_inner    local_inner    local_outer     global
	// _start          _start          _start           _end           _end          _end
	//

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

		global_array_.global_start_= global_start_;
		global_array_.global_count_= global_count_;

		Decompose(1,0,0);
	}

	void Decompose(unsigned int num_process=0,unsigned int process_num=0,unsigned int ghost_width=0)
	{
		if(num_process<=1)
		{
			num_process=GLOBAL_COMM.GetSize();
			process_num=GLOBAL_COMM.GetRank();
		}
		global_array_.Decompose(num_process,process_num,ghost_width);

		local_inner_start_=global_array_.local_.inner_start;
		local_inner_count_=global_array_.local_.inner_count;
		local_outer_start_=global_array_.local_.outer_start;
		local_outer_count_=global_array_.local_.outer_count;

		UpdateHash();
	}

	template<typename TV,typename ... Args>
	void UpdateGhosts(TV pdata,Args const &... args)const
	{
		global_array_.UpdateGhosts(&(*pdata),std::forward<Args const &>(args)...);
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

		local_outer_start_index_= Compact(local_outer_start_);
		local_outer_end_index_= Compact(local_outer_start_+local_outer_count_);
	}

	inline size_type Hash(iterator s) const
	{
		auto d =( Decompact(s.self_ ) >> D_FP_POS)-local_outer_start_+local_outer_count_;

		size_type res =

		((d[0] )%local_outer_count_[0]) * hash_stride_[0] +

		((d[1] )%local_outer_count_[1]) * hash_stride_[1] +

		((d[2] )%local_outer_count_[2]) * hash_stride_[2];

		switch (NodeId(s.self_))
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

	size_type GetNumOfElements(int IFORM = VERTEX) const
	{
		return global_count_[0] * global_count_[1] * global_count_[2] * ((IFORM == VERTEX || IFORM == VOLUME) ? 1 : 3);
	}

	nTuple<NDIMS, size_type> const& GetLocalDimensions() const
	{
		return local_outer_count_;
	}
	size_type GetLocalNumOfElements(int IFORM = VERTEX) const
	{
		return local_outer_count_[0] * local_outer_count_[1] * local_outer_count_[2]
		* ((IFORM == VERTEX || IFORM == VOLUME) ? 1 : 3);
	}
	size_type GetLocalMemorySize(int IFORM = VERTEX,int ele_size=1) const
	{
		return local_outer_count_[0] * local_outer_count_[1] * local_outer_count_[2]
		* ((IFORM == VERTEX || IFORM == VOLUME) ? 1 : 3)*ele_size;
	}
	int GetDataSetShape(int IFORM, size_type * global_start = nullptr, size_type * global_count = nullptr, size_type * local_outer_start = nullptr,
	size_type * local_outer_count = nullptr, size_type * local_inner_start = nullptr, size_type * local_inner_count = nullptr ) const
	{
		int rank = 0;

		for (int i = 0; i < NDIMS; ++i)
		{
			if ( global_count_[i] > 1)
			{

				if (global_start != nullptr)
				global_start[rank] = global_start_[i];

				if (global_count != nullptr)
				global_count[rank] = global_count_[i];

				if (local_outer_start != nullptr)
				local_outer_start[rank] = local_inner_start_[i];

				if (local_outer_count != nullptr)
				local_outer_count[rank] = local_outer_count_[i];

				if (local_inner_start != nullptr)
				local_inner_start[rank] = local_inner_start_[i];

				if (local_inner_count != nullptr)
				local_inner_count[rank] = local_inner_count_[i];

				++rank;
			}

		}
		if (IFORM == EDGE || IFORM == FACE)
		{
			if (global_start != nullptr)
			global_start[rank] = 0;

			if (global_count != nullptr)
			global_count[rank] = 3;

			if (local_outer_start != nullptr)
			local_outer_start[rank] = 0;

			if (local_outer_count != nullptr)
			local_outer_count[rank] = 3;

			if (local_inner_start != nullptr)
			local_inner_start[rank] = 0;

			if (local_inner_count != nullptr)
			local_inner_count[rank] = 3;

			++rank;
		}
		return rank;
	}

	range GetRange(int IFORM = VERTEX) const
	{
		return GetRange(global_start_, global_count_, IFORM);
	}

	range GetRange(nTuple<NDIMS,size_t>const& start,nTuple<NDIMS,size_t>const& count,int IFORM = VERTEX) const
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

		return range(start, count, shift);
	}

	template<int I>
	inline int GetAdjacentCells(Int2Type<I>, Int2Type<I>, iterator s, iterator *v) const
	{
		v[0] = s;
		return 1;
	}

	inline int GetAdjacentCells(Int2Type<EDGE>, Int2Type<VERTEX>, iterator s, iterator *v) const
	{
		v[0] = s + DeltaIndex(s.self_);
		v[1] = s - DeltaIndex(s.self_);
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

		auto di = DeltaIndex(Roate(Dual(s.self_)));
		auto dj = DeltaIndex(InverseRoate(Dual(s.self_)));

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
		auto di = _DI >> (HeightOfTree(s.self_) + 1);
		auto dj = _DJ >> (HeightOfTree(s.self_) + 1);
		auto dk = _DK >> (HeightOfTree(s.self_) + 1);

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

		auto di = _DI >> (HeightOfTree(s.self_) + 1);
		auto dj = _DJ >> (HeightOfTree(s.self_) + 1);
		auto dk = _DK >> (HeightOfTree(s.self_) + 1);

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
		auto d1 = DeltaIndex(Roate(Dual(s.self_)));
		auto d2 = DeltaIndex(InverseRoate(Dual(s.self_)));
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
		auto di = _DI >> (HeightOfTree(s.self_) + 1);
		auto dj = _DJ >> (HeightOfTree(s.self_) + 1);
		auto dk = _DK >> (HeightOfTree(s.self_) + 1);

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
		auto di = _DI >> (HeightOfTree(s.self_) + 1);
		auto dj = _DJ >> (HeightOfTree(s.self_) + 1);
		auto dk = _DK >> (HeightOfTree(s.self_) + 1);

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

		auto d1 = DeltaIndex(Roate((s.self_)));
		auto d2 = DeltaIndex(InverseRoate((s.self_)));

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

		auto di = _DI >> (HeightOfTree(s.self_) + 1);
		auto dj = _DJ >> (HeightOfTree(s.self_) + 1);
		auto dk = _DK >> (HeightOfTree(s.self_) + 1);

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

		auto di = _DI >> (HeightOfTree(s.self_) + 1);
		auto dj = _DJ >> (HeightOfTree(s.self_) + 1);
		auto dk = _DK >> (HeightOfTree(s.self_) + 1);

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

		auto d1 = DeltaIndex(Roate((s.self_)));
		auto d2 = DeltaIndex(InverseRoate((s.self_)));

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

		auto d = DeltaIndex(Dual(s.self_));
		v[0] = s + d;
		v[1] = s - d;

		return 2;
	}

	//***************************************************************************************************
	//* Auxiliary functions
	//***************************************************************************************************

	static compact_index_type Dual(compact_index_type r)
	{

		return (r & (~(_DA >> (HeightOfTree(r) + 1))))
		| ((~(r & (_DA >> (HeightOfTree(r) + 1)))) & (_DA >> (HeightOfTree(r) + 1)));

	}

	static unsigned int NodeId(compact_index_type r)
	{
		auto s = (r & (_DA >> (HeightOfTree(r) + 1))) >> (D_FP_POS - HeightOfTree(r) - 1);

		return ((s >> (INDEX_DIGITS * 2)) | (s >> (INDEX_DIGITS - 1)) | (s << 2UL)) & (7UL);
	}

	static unsigned int HeightOfTree(compact_index_type r)
	{
		return r >> (INDEX_DIGITS * 3);
	}
	static compact_index_type Roate(compact_index_type r)
	{

		compact_index_type res;

		res = r & (~(_DA >> (HeightOfTree(r) + 1)));

		res |= ((r & ((_DI | _DJ) >> (HeightOfTree(r) + 1))) >> INDEX_DIGITS) |

		((r & (_DK >> (HeightOfTree(r) + 1))) << (INDEX_DIGITS * 2))

		;
		return res;

	}

	/**
	 *  rotate vector direction  mask
	 *  (1/2,0,0) => (0,0,1/2) or   (1/2,1/2,0) => (1/2,0,1/2)
	 * @param s
	 * @return
	 */
	static compact_index_type InverseRoate(compact_index_type r)
	{
		compact_index_type res;

		res = r & ~(_DA >> (HeightOfTree(r) + 1));

		res |= ((r & (_DI >> (HeightOfTree(r) + 1))) >> (INDEX_DIGITS * 2)) |

		((r & (_DJ >> (HeightOfTree(r) + 1))) << INDEX_DIGITS) |

		((r & (_DK >> (HeightOfTree(r) + 1))) << INDEX_DIGITS);

		return res;
	}
	static compact_index_type DeltaIndex(compact_index_type r)
	{
		return (r & (_DA >> (HeightOfTree(r) + 1)));
	}

	static compact_index_type DeltaIndex(unsigned int i,compact_index_type r =0UL)
	{
		return (1UL << (INDEX_DIGITS * (NDIMS - i - 1) + D_FP_POS - HeightOfTree(r) - 1));
	}

	//! get the direction of vector(edge) 0=>x 1=>y 2=>z
	static compact_index_type DirectionOfVector(compact_index_type r)
	{
		compact_index_type s = (r & (_DA >> (HeightOfTree(r) + 1))) >> (D_FP_POS - HeightOfTree(r) - 1);

		return ((s >> (INDEX_DIGITS * 2)) | (s >> (INDEX_DIGITS - 1)) | (s << 2UL)) & (7UL);
	}

	/**
	 * Get component number or vector direction
	 * @param s
	 * @return
	 */
	static size_type ComponentNum(compact_index_type r)
	{
		size_type res = 0;
		switch (NodeId(r))
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

	static size_type IForm(compact_index_type r)
	{
		size_type res = 0;
		switch (NodeId(r))
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

	//****************************************************************************************************
	//iterator
	//****************************************************************************************************

	struct iterator
	{
/// One of the @link iterator_tags tag types@endlink.
		typedef std::bidirectional_iterator_tag iterator_category;

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

		iterator(iterator const & r)
		: self_(r.self_),start_(r.start_),end_(r.end_)
		{
		}
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
		bool operator<(iterator const & rhs) const
		{
			return (self_<rhs.self_);
		}
		iterator const & operator*() const
		{
			return *this;
		}

		iterator const* operator ->() const
		{
			return this;
		}

		bool isNull()const
		{
			return self_==0UL;
		}
		void NextCell()
		{
			if(self_!=end_)
			{
				auto D = (1UL << (D_FP_POS - HeightOfTree(self_)));

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
			else
			{
				self_=0UL;
			}

		}

		void PreviousCell()
		{
			if(self_!=start_)
			{
				auto D = (1UL << (D_FP_POS - HeightOfTree(self_)));

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
			else
			{
				self_=0UL;
			}

		}

		iterator & operator ++()
		{
			auto n = NodeId(self_);

			if (n == 0 || n == 4 || n == 3 || n == 7)
			{
				NextCell();
			}

			self_ = OcForest::Roate(self_);

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

			auto n = NodeId(self_);

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

		nTuple<NDIMS,size_type> Decompact()const
		{
			return (OcForest::Decompact(self_)>>D_FP_POS)- (OcForest::Decompact(start_)>>D_FP_POS);
		}

	}; // class iterator

	struct range
	{
	public:
		typedef typename OcForest::iterator iterator;
		typedef iterator value_type;

		nTuple<NDIMS, size_type> start_;

		nTuple<NDIMS, size_type> count_;

		compact_index_type shift_ = 0UL;

		range():shift_(0UL)
		{
		}

		range(range const & r ):start_(r.start_),count_(r.count_),shift_(r.shift_)
		{
		}
		range(nTuple<NDIMS, size_type> const & start, nTuple<NDIMS, size_type> const& count,
		compact_index_type node_shift = 0UL)
		: start_(start), count_(count), shift_(node_shift)
		{
		}

		~range()
		{
		}

		iterator begin() const
		{
			return iterator((Compact(start_) << D_FP_POS) | shift_, ((Compact(start_) << D_FP_POS) | shift_),
			((Compact(start_ + count_) << D_FP_POS) | shift_));
		}
		iterator end() const
		{
			iterator res(begin());

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

		iterator rbegin() const
		{
			iterator res(rend());

			if (count_[0] * count_[1] * count_[2] > 0)
			{
				res = iterator(

				(Compact(start_ + count_ ) << D_FP_POS) | shift_,

				((Compact(start_) << D_FP_POS) | shift_),

				((Compact(start_ + count_) << D_FP_POS) | shift_)

				);
			}
			return res;
		}
		iterator rend() const
		{
			auto res=begin();
			res.PreviousCell();
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
		range Split(unsigned int num_process, unsigned int process_num, unsigned int ghost_width = 0) const
		{
			int n=0;
			size_type L=0;
			for (int i = 0; i < NDIMS; ++i)
			{
				if(count_[i]>L)
				{
					L=count_[i];
					n=i;
				}
			}

			nTuple<NDIMS,size_type> start,count;

			count = count_;
			start = start_;

			if ((2 * ghost_width * num_process > count_[n] || num_process > count_[n]) )
			{
				if( process_num>0) count=0;
			}
			else
			{
				start[n] += (count_[n] * process_num ) / num_process;
				count[n]= (count_[n] * (process_num + 1)) / num_process -(count_[n] * process_num ) / num_process;
			}

			return range(start,count,shift_);
		}
	};
	typedef range Range;
	typedef range const_range;
	// class Range

	/***************************************************************************************************
	 *
	 *  Geomertry dependence
	 *
	 *  INDEX_ZERO <-> Coordinates Zero
	 *
	 */

	nTuple<NDIMS, Real> GetExtents() const
	{

		nTuple<NDIMS, Real> res;

		for (int i = 0; i < NDIMS; ++i )
		{
			res[i]=global_count_[i ]>1?static_cast<Real>(global_count_[i ]):0.0;
		}

		return res;
	}

	//***************************************************************************************************
	// Coordinates
	inline coordinates_type GetCoordinates(iterator const& s) const
	{
		auto d = Decompact(s.self_)-(global_start_<<D_FP_POS);

		return coordinates_type(
		{
			static_cast<Real>(d[0] )*R_DX ,
			static_cast<Real>(d[1] )*R_DX ,
			static_cast<Real>(d[2] )*R_DX ,
		});
	}

	coordinates_type CoordinatesLocalToGlobal(iterator const& s, coordinates_type r) const
	{
		return GetCoordinates(s) + r * static_cast<Real>(1UL << (D_FP_POS - HeightOfTree(s.self_)));
	}

	inline iterator CoordinatesGlobalToLocalDual(coordinates_type *px, compact_index_type shift = 0UL) const
	{
		return CoordinatesGlobalToLocal(px, shift);
	}
	const Real zero= ((((1UL << (INDEX_DIGITS - D_FP_POS - 1)) - 1) << D_FP_POS));
	const Real dx= (1UL << (D_FP_POS ));
	const Real inv_dx= 1.0/dx;

	compact_index_type local_outer_start_index_= 0UL;
	compact_index_type local_outer_end_index_= 0UL;
	inline iterator CoordinatesGlobalToLocal(coordinates_type *px, compact_index_type shift = 0UL) const
	{
		auto & x = *px;

		x = x*dx +zero;

		nTuple<NDIMS, size_type> idx;

		idx = x;

		idx -= Decompact(shift);
		idx = idx >> (D_FP_POS );
		idx = idx << (D_FP_POS );

//		x[0] = (x[0] - static_cast<Real>(idx[0]))*inv_dx;
//
//		x[1] = (x[1] - static_cast<Real>(idx[1]))*inv_dx;
//
//		x[2] = (x[2] - static_cast<Real>(idx[2]))*inv_dx;

		x=(x-idx)*inv_dx;

		return iterator(

		Compact(idx) | shift,

		local_outer_start_index_ | shift,

		local_outer_end_index_ | shift

		);

	}

	static Real Volume(iterator s)
	{
//		static constexpr double volume_[8][D_FP_POS] =
//		{
//
//			1, 1, 1, 1, // 000
//
//			1, 1.0 / 2, 1.0 / 4, 1.0 / 8,// 001
//
//			1, 1.0 / 2, 1.0 / 4, 1.0 / 8,// 010
//
//			1, 1.0 / 4, 1.0 / 16, 1.0 / 64,// 011
//
//			1, 1.0 / 2, 1.0 / 4, 1.0 / 8,// 100
//
//			1, 1.0 / 4, 1.0 / 16, 1.0 / 64,// 101
//
//			1, 1.0 / 4, 1.0 / 16, 1.0 / 64,// 110
//
//			1, 1.0 / 8, 1.0 / 64, 1.0 / 512// 111
//
//		};
//		return volume_[NodeId(s.self_)][HeightOfTree(s.self_)];

		return 1.0;
	}

	static Real InvVolume(iterator s)
	{
//		static constexpr double inv_volume_[8][D_FP_POS] =
//		{
//
//			1, 1, 1, 1, // 000
//
//			1, 2, 4, 8,// 001
//
//			1, 2, 4, 8,// 010
//
//			1, 4, 16, 64,// 011
//
//			1, 2, 4, 8,// 100
//
//			1, 4, 16, 64,// 101
//
//			1, 4, 16, 64,// 110
//
//			1, 8, 64, 512// 111
//
//		};
//		return inv_volume_[NodeId(s.self_)][HeightOfTree(s.self_)];
		return 1.0;
	}

//	static Real Volume(iterator s)
//	{
//		static constexpr double volume_[8][D_FP_POS] =
//		{
//
//			1, 1, 1, 1, // 000
//
//			8, 4, 2, 1,// 001
//
//			8, 4, 2, 1,// 010
//
//			64, 16, 4, 1,// 011
//
//			8, 4, 2, 1,// 100
//
//			64, 16, 4, 1,// 101
//
//			64, 16, 4, 1,// 110
//
//			128, 32, 8, 1// 111
//
//		};
//
//		return volume_[NodeId(s.self_)][HeightOfTree(s.self_)];
//	}
//
//	static Real InvVolume(iterator s)
//	{
//		static constexpr double inv_volume_[8][D_FP_POS] =
//		{
//
//			1, 1, 1, 1, // 000
//
//			1.0 / 8, 1.0 / 4, 1.0 / 2, 1.0,// 001
//
//			1.0 / 8, 1.0 / 4, 1.0 / 2, 1.0,// 010
//
//			1.0 / 64, 1.0 / 16, 1.0 / 4, 1.0,// 011
//
//			1.0 / 8, 1.0 / 4, 1.0 / 2, 1.0,// 100
//
//			1.0 / 64, 1.0 / 16, 1.0 / 4, 1.0,// 101
//
//			1.0 / 64, 1.0 / 16, 1.0 / 4, 1.0,// 110
//
//			1.0 / 128, 1.0 / 32, 1.0 / 8, 1.0// 111
//
//		};
//
//		return inv_volume_[NodeId(s.self_)][HeightOfTree(s.self_)];
//	}

	static Real InvDualVolume(iterator s)
	{
		return InvVolume(Dual(s.self_));
	}
	static Real DualVolume(iterator s)
	{
		return Volume(Dual(s.self_));
	}
	//***************************************************************************************************

};
// class OcForest

}
// namespace simpla

#endif /* OCTREE_FOREST_H_ */
