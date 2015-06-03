/*
 * octree_forest.h
 *
 *  created on: 2014-2-21
 *      Author: salmon
 */

#ifndef OCTREE_FOREST_H_
#define OCTREE_FOREST_H_

#include <algorithm>
#include <cassert>
#include <cmath>

#include <limits>
#include <thread>
#include <iterator>
#include "../gtl/ntuple.h"
#include "../utilities/primitives.h"
#include "../utilities/type_traits.h"
#include "../utilities/pretty_stream.h"
#include "../design_pattern/memory_pool.h"
#include "../parallel/distributed_array.h"

namespace simpla
{

struct OcForest
{

	typedef OcForest this_type;
	static constexpr  unsigned int  MAX_NUM_NEIGHBOUR_ELEMENT = 12;
	static constexpr  unsigned int  MAX_NUM_VERTEX_PER_CEL = 8;
	static constexpr  unsigned int  NDIMS = 3;

	typedef long index_type;

	typedef unsigned long index_type;

	struct iterator;

	struct range;

	typedef range range_type;

	typedef nTuple<NDIMS, Real> coordinate_type;

	typedef std::map<iterator, nTuple<3, coordinate_type>> surface_type;

	//!< signed long is 63bit, unsigned long is 64 bit, add a sign bit
	static constexpr   unsigned int   FULL_DIGITS = std::numeric_limits<index_type>::digits;

	static constexpr   unsigned int   D_FP_POS = 4; //!< default floating-point position

	static constexpr   unsigned int   INDEX_DIGITS = (FULL_DIGITS - CountBits<D_FP_POS>::n) / 3;

	static constexpr index_type INDEX_MASK = (1UL << INDEX_DIGITS) - 1;
	static constexpr index_type TREE_ROOT_MASK = ((1UL << (INDEX_DIGITS - D_FP_POS)) - 1) << D_FP_POS;
	static constexpr index_type ROOT_MASK = TREE_ROOT_MASK | (TREE_ROOT_MASK << INDEX_DIGITS)
	        | (TREE_ROOT_MASK << (INDEX_DIGITS * 2));

	static constexpr index_type INDEX_ZERO = (((1UL << (INDEX_DIGITS - D_FP_POS - 1)) - 1));

	static constexpr index_type COMPACT_INDEX_ZERO = (INDEX_ZERO << D_FP_POS);

	static constexpr index_type FP_POS = 1UL << D_FP_POS;

	static constexpr Real R_FP_POS = static_cast<Real>(FP_POS);

	static constexpr Real R_INV_FP_POS = 1.0 / R_FP_POS;

	nTuple<NDIMS, Real> R_INV_DX;
	nTuple<NDIMS, Real> R_DX;
	//***************************************************************************************************

	static constexpr index_type NO_HEAD_FLAG = ~((~0UL) << (INDEX_DIGITS * 3));
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

	static constexpr index_type _DI = 1UL << (D_FP_POS + 2 * INDEX_DIGITS);
	static constexpr index_type _DJ = 1UL << (D_FP_POS + INDEX_DIGITS);
	static constexpr index_type _DK = 1UL << (D_FP_POS);
	static constexpr index_type _DA = _DI | _DJ | _DK;

	//mask of direction
	static constexpr index_type _MI = ((1UL << (INDEX_DIGITS)) - 1) << (INDEX_DIGITS * 2);
	static constexpr index_type _MJ = ((1UL << (INDEX_DIGITS)) - 1) << (INDEX_DIGITS);
	static constexpr index_type _MK = ((1UL << (INDEX_DIGITS)) - 1);
	static constexpr index_type _MH = ((1UL << (FULL_DIGITS - INDEX_DIGITS * 3 + 1)) - 1)
	        << (INDEX_DIGITS * 3 + 1);

	// mask of sub-tree
	static constexpr index_type _MTI = ((1UL << (D_FP_POS)) - 1) << (INDEX_DIGITS * 2);
	static constexpr index_type _MTJ = ((1UL << (D_FP_POS)) - 1) << (INDEX_DIGITS);
	static constexpr index_type _MTK = ((1UL << (D_FP_POS)) - 1);

	// mask of root
	static constexpr index_type _MRI = _MI & (~_MTI);
	static constexpr index_type _MRJ = _MJ & (~_MTJ);
	static constexpr index_type _MRK = _MK & (~_MTK);

	static index_type compact(nTuple<NDIMS, index_type> const & idx, index_type shift = 0UL)
	{
		return

		((static_cast<index_type>(idx[0] * FP_POS + COMPACT_INDEX_ZERO) & INDEX_MASK) << (INDEX_DIGITS * 2)) |

		((static_cast<index_type>(idx[1] * FP_POS + COMPACT_INDEX_ZERO) & INDEX_MASK) << (INDEX_DIGITS)) |

		((static_cast<index_type>(idx[2] * FP_POS + COMPACT_INDEX_ZERO) & INDEX_MASK)) |

		shift;
	}
	static nTuple<NDIMS, index_type> decompact(index_type s)
	{
		return nTuple<NDIMS, index_type>( {

		static_cast<index_type>((s >> (INDEX_DIGITS * 2)) & INDEX_MASK) - COMPACT_INDEX_ZERO,

		static_cast<index_type>((s >> (INDEX_DIGITS)) & INDEX_MASK) - COMPACT_INDEX_ZERO,

		static_cast<index_type>(s & INDEX_MASK) - COMPACT_INDEX_ZERO

		});
	}
	static nTuple<NDIMS, index_type> DecompactRoot(index_type s)
	{
		return nTuple<NDIMS, index_type>( {

		static_cast<index_type>((s >> (INDEX_DIGITS * 2 + D_FP_POS)) & INDEX_MASK) - INDEX_ZERO,

		static_cast<index_type>((s >> (INDEX_DIGITS + D_FP_POS)) & INDEX_MASK) - INDEX_ZERO,

		static_cast<index_type>((s >> D_FP_POS) & INDEX_MASK) - INDEX_ZERO

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

	template<typename TV> using Container=std::shared_ptr<TV>;

	template<unsigned int iform, typename TV> inline std::shared_ptr<TV> MakeContainer() const
	{
		return (MEMPOOL.allocate_shared_ptr < TV > (get_local_num_of_elements(iform)));
	}

	template<typename TDict, typename ...Others>
	void load(TDict const & dict, Others const& ...)
	{
		try
		{
			LOGGER << "Load OcForest ";
			set_dimensions(dict["Dimensions"].template as<nTuple<3, index_type>>());
		}
		catch(...)
		{
			PARSER_ERROR("Configure OcForest error!");
		}
	}

	std::string save(std::string const &path) const
	{
		std::stringstream os;

		os << "\tDimensions =  " << get_dimensions();

		return os.str();
	}

	unsigned long clock_ = 0UL;

	void next_timestep()
	{
		++clock_;
	}
	unsigned long get_clock() const
	{
		return clock_;
	}

	//***************************************************************************************************
	// Local Data Set

	nTuple<NDIMS, index_type> global_start_, global_count_;

	nTuple<NDIMS, index_type> local_outer_start_, local_outer_count_;

	nTuple<NDIMS, index_type> local_inner_start_, local_inner_count_;

	nTuple<NDIMS, index_type> hash_stride_;

	enum
	{
		FAST_FIRST, SLOW_FIRST
	};

	int array_order_ = SLOW_FIRST;

	index_type global_start_index_= 0UL;
	index_type local_outer_start_index_= 0UL;
	index_type local_outer_end_index_= 0UL;

	DistributedArray<NDIMS> global_array_;
	//
	//   |----------------|----------------|---------------|--------------|------------|
	//   ^                ^                ^               ^              ^            ^
	//   |                |                |               |              |            |
	//global          local_outer      local_inner    local_inner    local_outer     global
	// _start          _start          _start           _end           _end          _end
	//

	void set_dimensions( )
	{
	}

	template<typename TI>
	void set_dimensions(TI const &d)
	{
		for (int i = 0; i < NDIMS; ++i)
		{
			index_type length = d[i] > 0 ? d[i] : 1;

			ASSERT(length<COMPACT_INDEX_ZERO );

			global_start_[i] = 0;
			global_count_[i] = length;

			if(global_count_[i] >1)
			{
				R_INV_DX[i]=static_cast<Real>(length);
				R_DX[i]=1.0/R_INV_DX[i];
			}
			else
			{
				R_INV_DX[i]=0;
				R_DX[i]=0;
			}
		}

		global_array_.global_start_= global_start_;
		global_array_.global_count_= global_count_;

		Decompose(1,0,0);
	}

	void Decompose(  unsigned int   num_process=0,  unsigned int   process_num=0,  unsigned int   ghost_width=0)
	{
		if(num_process<=1)
		{
			num_process=GLOBAL_COMM.get_size();
			process_num=GLOBAL_COMM.get_rank();
		}
		global_array_.Decompose(num_process,process_num,ghost_width);

		local_inner_start_=global_array_.local_.inner_start;
		local_inner_count_=global_array_.local_.inner_count;
		local_outer_start_=global_array_.local_.outer_start;
		local_outer_count_=global_array_.local_.outer_count;

		updateHash();
	}

	void updateHash()
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
		global_start_index_=compact(global_start_)<<D_FP_POS;
		local_outer_start_index_= compact(local_outer_start_)<<D_FP_POS;
		local_outer_end_index_= compact(local_outer_start_+local_outer_count_)<<D_FP_POS;
	}

	inline index_type Hash(index_type s) const
	{
		auto d =( decompact(s ) >> D_FP_POS)-local_outer_start_+local_outer_count_;

		index_type res =

		((d[0] )%local_outer_count_[0]) * hash_stride_[0] +

		((d[1] )%local_outer_count_[1]) * hash_stride_[1] +

		((d[2] )%local_outer_count_[2]) * hash_stride_[2];

		switch (ele_suffix(s))
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

	nTuple<NDIMS, index_type> const& get_dimensions() const
	{
		return global_count_;
	}

	index_type get_num_of_elements(int IFORM = VERTEX) const
	{
		return global_count_[0] * global_count_[1] * global_count_[2] * ((IFORM == VERTEX || IFORM == VOLUME) ? 1 : 3);
	}

	nTuple<NDIMS, index_type> const& get_local_dimensions() const
	{
		return local_outer_count_;
	}
	index_type get_local_num_of_elements(int IFORM = VERTEX) const
	{
		return local_outer_count_[0] * local_outer_count_[1] * local_outer_count_[2]
		* ((IFORM == VERTEX || IFORM == VOLUME) ? 1 : 3);
	}
	index_type get_local_memory_size(int IFORM = VERTEX,int ele_size=1) const
	{
		return local_outer_count_[0] * local_outer_count_[1] * local_outer_count_[2]
		* ((IFORM == VERTEX || IFORM == VOLUME) ? 1 : 3)*ele_size;
	}
	int get_dataset_shape(int IFORM, size_t * global_start = nullptr, size_t * global_count = nullptr, size_t * local_outer_start = nullptr,
	size_t * local_outer_count = nullptr, size_t * local_inner_start = nullptr, size_t * local_inner_count = nullptr ) const
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
	static index_type get_shift(  unsigned int   nodeid,index_type h=0UL)
	{
		index_type shift = h << (INDEX_DIGITS * 3);

		return

		(((nodeid>>2) & 1UL)<<(INDEX_DIGITS*2+D_FP_POS-h-1)) |

		(((nodeid>>1) & 1UL)<<(INDEX_DIGITS+D_FP_POS-h-1)) |

		((nodeid & 1UL)<<( D_FP_POS-h-1))
		;
	}

	static index_type get_first_node_shift(int iform)
	{
		index_type res;
		switch(iform)
		{
			case VERTEX:
			res=0;
			break;
			case EDGE:
			res=4;
			break;
			case FACE:
			res=3;
			break;
			case VOLUME:
			res=7;
			break;
		}
		return get_shift(res);
	}

	inline   unsigned int   get_vertices( index_type s, index_type *v) const
	{
		  unsigned int   n=0;
		switch(IForm(s))
		{
			case VERTEX:
			{
				v[0]=s;
			}
			n=1;
			break;
			case EDGE:
			{
				auto di=delta_index(s);
				v[0] = s + di;
				v[1] = s - di;
			}
			n=2;
			break;

			case FACE:
			{
				auto di = delta_index(rotate(dual(s)));
				auto dj = delta_index(inverse_rotate(dual(s)));

				v[0] = s - di - dj;
				v[1] = s - di - dj;
				v[2] = s + di + dj;
				v[3] = s + di + dj;
				n=4;
			}
			break;
			case VOLUME:
			{
				auto di = _DI >> (HeightOfTree(s) + 1);
				auto dj = _DJ >> (HeightOfTree(s) + 1);
				auto dk = _DK >> (HeightOfTree(s) + 1);

				v[0] = ((s - di) - dj) - dk;
				v[1] = ((s - di) - dj) + dk;
				v[2] = ((s - di) + dj) - dk;
				v[3] = ((s - di) + dj) + dk;

				v[4] = ((s + di) - dj) - dk;
				v[5] = ((s + di) - dj) + dk;
				v[6] = ((s + di) + dj) - dk;
				v[7] = ((s + di) + dj) + dk;
				n=8;
			}
			break;
		}
		return n;
	}

	template<unsigned int I>
	inline  unsigned int  get_adjacent_cells(std::integral_constant<unsigned int ,I>, std::integral_constant<unsigned int ,I>, index_type s, index_type *v) const
	{
		v[0] = s;
		return 1;
	}

	inline  unsigned int  get_adjacent_cells(std::integral_constant<unsigned int ,EDGE>, std::integral_constant<unsigned int ,VERTEX>, index_type s, index_type *v) const
	{
		v[0] = s + delta_index(s);
		v[1] = s - delta_index(s);
		return 2;
	}

	inline  unsigned int  get_adjacent_cells(std::integral_constant<unsigned int ,FACE>, std::integral_constant<unsigned int ,VERTEX>, index_type s, index_type *v) const
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

		auto di = delta_index(rotate(dual(s)));
		auto dj = delta_index(inverse_rotate(dual(s)));

		v[0] = s - di - dj;
		v[1] = s - di - dj;
		v[2] = s + di + dj;
		v[3] = s + di + dj;

		return 4;
	}

	inline  unsigned int  get_adjacent_cells(std::integral_constant<unsigned int ,VOLUME>, std::integral_constant<unsigned int ,VERTEX>, index_type s, index_type *v) const
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
		auto di = _DI >> (HeightOfTree(s) + 1);
		auto dj = _DJ >> (HeightOfTree(s) + 1);
		auto dk = _DK >> (HeightOfTree(s) + 1);

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

	inline  unsigned int  get_adjacent_cells(std::integral_constant<unsigned int ,VERTEX>, std::integral_constant<unsigned int ,EDGE>, index_type s, index_type *v) const
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

		auto di = _DI >> (HeightOfTree(s) + 1);
		auto dj = _DJ >> (HeightOfTree(s) + 1);
		auto dk = _DK >> (HeightOfTree(s) + 1);

		v[0] = s + di;
		v[1] = s - di;

		v[2] = s + dj;
		v[3] = s - dj;

		v[4] = s + dk;
		v[5] = s - dk;

		return 6;
	}

	inline  unsigned int  get_adjacent_cells(std::integral_constant<unsigned int ,FACE>, std::integral_constant<unsigned int ,EDGE>, index_type s, index_type *v) const
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
		auto d1 = delta_index(rotate(dual(s)));
		auto d2 = delta_index(inverse_rotate(dual(s)));
		v[0] = s - d1;
		v[1] = s + d1;
		v[2] = s - d2;
		v[3] = s + d2;

		return 4;
	}

	inline  unsigned int  get_adjacent_cells(std::integral_constant<unsigned int ,VOLUME>, std::integral_constant<unsigned int ,EDGE>, index_type s, index_type *v) const
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
		auto di = _DI >> (HeightOfTree(s) + 1);
		auto dj = _DJ >> (HeightOfTree(s) + 1);
		auto dk = _DK >> (HeightOfTree(s) + 1);

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

	inline  unsigned int  get_adjacent_cells(std::integral_constant<unsigned int ,VERTEX>, std::integral_constant<unsigned int ,FACE>, index_type s, index_type *v) const
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
		auto di = _DI >> (HeightOfTree(s) + 1);
		auto dj = _DJ >> (HeightOfTree(s) + 1);
		auto dk = _DK >> (HeightOfTree(s) + 1);

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

	inline  unsigned int  get_adjacent_cells(std::integral_constant<unsigned int ,EDGE>, std::integral_constant<unsigned int ,FACE>, index_type s, index_type *v) const
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

		auto d1 = delta_index(rotate((s)));
		auto d2 = delta_index(inverse_rotate((s)));

		v[0] = s - d1;
		v[1] = s + d1;
		v[2] = s - d2;
		v[3] = s + d2;

		return 4;
	}

	inline  unsigned int  get_adjacent_cells(std::integral_constant<unsigned int ,VOLUME>, std::integral_constant<unsigned int ,FACE>, index_type s, index_type *v) const
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

		auto di = _DI >> (HeightOfTree(s) + 1);
		auto dj = _DJ >> (HeightOfTree(s) + 1);
		auto dk = _DK >> (HeightOfTree(s) + 1);

		v[0] = s - di;
		v[1] = s + di;

		v[2] = s - di;
		v[3] = s + dj;

		v[4] = s - dk;
		v[5] = s + dk;

		return 6;
	}

	inline  unsigned int  get_adjacent_cells(std::integral_constant<unsigned int ,VERTEX>, std::integral_constant<unsigned int ,VOLUME>, index_type s, index_type *v) const
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

		auto di = _DI >> (HeightOfTree(s) + 1);
		auto dj = _DJ >> (HeightOfTree(s) + 1);
		auto dk = _DK >> (HeightOfTree(s) + 1);

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

	inline  unsigned int  get_adjacent_cells(std::integral_constant<unsigned int ,EDGE>, std::integral_constant<unsigned int ,VOLUME>, index_type s, index_type *v) const
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

		auto d1 = delta_index(rotate((s)));
		auto d2 = delta_index(inverse_rotate((s)));

		v[0] = s - d1 - d2;
		v[1] = s + d1 - d2;
		v[2] = s - d1 + d2;
		v[3] = s + d1 + d2;
		return 4;
	}

	inline  unsigned int  get_adjacent_cells(std::integral_constant<unsigned int ,FACE>, std::integral_constant<unsigned int ,VOLUME>, index_type s, index_type *v) const
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

		auto d = delta_index(dual(s));
		v[0] = s + d;
		v[1] = s - d;

		return 2;
	}

	//***************************************************************************************************
	//* Auxiliary functions
	//***************************************************************************************************

	static index_type dual(index_type r)
	{

		return (r & (~(_DA >> (HeightOfTree(r) + 1))))
		| ((~(r & (_DA >> (HeightOfTree(r) + 1)))) & (_DA >> (HeightOfTree(r) + 1)));

	}
	static   unsigned int   get_cell_index(index_type r)
	{
		index_type mask=(1UL<<(D_FP_POS-HeightOfTree(r)))-1;

		return r&(~(mask|(mask<<INDEX_DIGITS)|(mask<<(INDEX_DIGITS*2))));
	}
	static   unsigned int   ele_suffix(index_type r)
	{
		auto s = (r & (_DA >> (HeightOfTree(r) + 1))) >> (D_FP_POS - HeightOfTree(r) - 1);

		return ((s >> (INDEX_DIGITS * 2)) | (s >> (INDEX_DIGITS - 1)) | (s << 2UL)) & (7UL);
	}

	static   unsigned int   HeightOfTree(index_type r)
	{
		return r >> (INDEX_DIGITS * 3);
	}
	static index_type rotate(index_type r)
	{

		index_type res;

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
	static index_type inverse_rotate(index_type s)
	{
		index_type res;

		res = s & ~(_DA >> (HeightOfTree(s) + 1));

		res |= ((s & (_DI >> (HeightOfTree(s) + 1))) >> (INDEX_DIGITS * 2)) |

		((s & (_DJ >> (HeightOfTree(s) + 1))) << INDEX_DIGITS) |

		((s & (_DK >> (HeightOfTree(s) + 1))) << INDEX_DIGITS);

		return res;
	}
	static index_type delta_index(index_type r)
	{
		return (r & (_DA >> (HeightOfTree(r) + 1)));
	}

	static index_type delta_index(  unsigned int   i,index_type r )
	{
		return (1UL << (INDEX_DIGITS * (NDIMS - i - 1) + D_FP_POS - HeightOfTree(r) - 1))&r;
	}

	//! get the direction of vector(edge) 0=>x 1=>y 2=>z
	static index_type DirectionOfVector(index_type r)
	{
		index_type s = (r & (_DA >> (HeightOfTree(r) + 1))) >> (D_FP_POS - HeightOfTree(r) - 1);

		return ((s >> (INDEX_DIGITS * 2)) | (s >> (INDEX_DIGITS - 1)) | (s << 2UL)) & (7UL);
	}

	/**
	 * Get component number or vector direction
	 * @param s
	 * @return
	 */
	static index_type component_number(index_type s)
	{
		index_type res = 0;
		switch (ele_suffix(s))
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

	static index_type IForm(index_type r)
	{
		index_type res = 0;
		switch (ele_suffix(r))
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
		typedef std::bidirectional_iterator_tag iterator_category;

		typedef index_type value_type;

		typedef index_type difference_type;

		typedef value_type* pointer;

		typedef value_type& reference;

		index_type self_;

		index_type start_, end_;

		iterator(iterator const & r)
		: self_(r.self_),start_(r.start_),end_(r.end_)
		{
		}
		iterator(index_type s = 0, index_type b = 0, index_type e = 0)
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
		value_type const & operator*() const
		{
			return self_;
		}

		value_type const* operator ->() const
		{
			return &self_;
		}

		bool isNull()const
		{
			return self_==0UL;
		}
		void NextCell()
		{
			if(self_!=end_)
			{
				auto D = (1UL << (D_FP_POS ));

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
			auto n = ele_suffix(self_);

			if (n == 0 || n == 4 || n == 3 || n == 7)
			{
				NextCell();
			}

			self_ = rotate(self_);

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

			auto n = ele_suffix(self_);

			if (n == 0 || n == 1 || n == 6 || n == 7)
			{
				PreviousCell();
			}

			self_ = inverse_rotate(self_);

			return *this;
		}

		iterator operator --(int)
		{
			iterator res(*this);
			--res;
			return std::move(res);
		}

#define DEF_OP(_OP_)                                                                       \
				inline iterator & operator _OP_##=(index_type r)                           \
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
				inline iterator  operator _OP_(index_type const &r) const                 \
				{   iterator res(*this);                                                                               \
				   res=(( ( ((self_ _OP_ (r & _MI)) & _MI) |                              \
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

	}; // class iterator

	struct range
	{
	public:
		typedef typename OcForest::iterator iterator;
		typedef iterator value_type;

		nTuple<NDIMS, index_type> start_;

		nTuple<NDIMS, index_type> count_;

		  unsigned int   iform_=VERTEX;

		index_type shift_ = 0UL;

		range():shift_(get_shift(0))
		{
			for(int i=0;i<NDIMS;++i)
			{
				start_[i]=0;
				count_[i]=0;
			}
		}
		range(range const& r):start_(r.start_),count_(r.count_),iform_(r.iform_ ),shift_(r.shift_)
		{

		}
		range(range && r):start_(r.start_),count_(r.count_),iform_(r.iform_ ),shift_(r.shift_)
		{

		}
		range(int iform,range const & r ):start_(r.start_),count_(r.count_),iform_(iform ),shift_(get_first_node_shift(iform))
		{

		}
		range(  unsigned int   iform,nTuple<NDIMS, index_type> const & start, nTuple<NDIMS, index_type> const& count )
		: start_(start ), count_(count), iform_(iform),shift_(get_first_node_shift(iform))
		{

		}

		~range()
		{
		}

		iterator begin() const
		{
			return iterator((compact(start_,shift_) ) | shift_, ((compact(start_) ) | shift_),
			((compact(start_ + count_) ) | shift_));
		}
		iterator end() const
		{
			iterator res(begin());

			if (count_[0] * count_[1] * count_[2] > 0)
			{
				res = iterator(

				(compact(start_ + count_ - 1) ) | shift_,

				((compact(start_) ) | shift_),

				((compact(start_ + count_) ) | shift_)

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

				(compact(start_ + count_ ,shift_) ) ,

				(compact(start_,shift_)),

				(compact(start_ + count_,shift_) )

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

		nTuple<NDIMS, index_type> const& Extents() const
		{
			return count_;
		}
		index_type Size() const
		{
			return size();
		}
		index_type size() const
		{
			index_type n = 1;

			for (int i = 0; i < NDIMS; ++i)
			{
				n *= count_[i];
			}
			return n;
		}
		range split(  unsigned int   num_process,   unsigned int   process_num,   unsigned int   ghost_width = 0) const
		{
			int n=0;
			index_type L=0;
			for (int i = 0; i < NDIMS; ++i)
			{
				if(count_[i]>L)
				{
					L=count_[i];
					n=i;
				}
			}

			nTuple<NDIMS,index_type> start,count;

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

			return range(iform_,start,count );
		}

	};	// class Range

	template<typename T>
	range select(   unsigned int   iform, std::pair<T,T> domain)const
	{
		return select(iform,domain.first,domain.second);
	}

	range select(  unsigned int   iform, coordinate_type xmin, coordinate_type xmax)const
	{
		auto start=coordinates_to_index(&xmin,get_first_node_shift(iform))>>D_FP_POS;
		auto count=(coordinates_to_index(&xmax,get_first_node_shift(iform))>>D_FP_POS)- start+1;

		return select(iform,start,count);
	}

	range select(   unsigned int   iform, nTuple<NDIMS, index_type> start, nTuple<NDIMS, index_type> count)const
	{
		auto flag=Clipping( local_inner_start_, local_inner_count_, &start, &count);

		if (!flag)
		{
			start=local_inner_start_;
			count*=0;
		}

		return range( iform,start,count);
	}

	range select(  unsigned int   iform)const
	{
		return range(iform, local_inner_start_,local_inner_count_);
	}

	/***************************************************************************************************
	 *
	 *  Geomertry dependence
	 *
	 *  INDEX_ZERO <-> Coordinate Zero
	 *
	 */

	nTuple<NDIMS, Real> get_extents() const
	{

		nTuple<NDIMS, Real> res;

		for (int i = 0; i < NDIMS; ++i )
		{
			res[i]=global_count_[i ]>1?static_cast<Real>(global_count_[i ]):0.0;
		}

		return res;
	}

	//***************************************************************************************************
	// Coordinate
	inline coordinate_type get_coordinates(index_type s) const
	{

		auto d = decompact(s) - (global_start_<<D_FP_POS);

		return coordinate_type(
		{
			static_cast<Real>(d[0] )*R_DX[0]*R_INV_FP_POS ,
			static_cast<Real>(d[1] )*R_DX[1]*R_INV_FP_POS ,
			static_cast<Real>(d[2] )*R_DX[2]*R_INV_FP_POS

		});
	}

	coordinate_type coordinates_local_to_global(index_type s, coordinate_type r) const
	{
		auto d = decompact(s)-(global_start_<<D_FP_POS);
		Real scale=static_cast<Real>(1UL << (D_FP_POS - HeightOfTree(s)));
		coordinate_type res;

		for(int i=0;i<NDIMS;++i)
		{
			res[i]=(static_cast<Real>(d[i])+r[i]*scale)*R_DX[i]*R_INV_FP_POS;
		}
		return std::move(res);
	}

	inline index_type CoordinateGlobalToLocaldual(coordinate_type *px, index_type shift = 0UL) const
	{

		return (coordinates_global_to_local(px, dual(shift)));
	}

	inline nTuple<NDIMS,index_type> coordinates_to_index(coordinate_type *px, index_type shift = 0UL)const
	{
		auto & x = *px;

		nTuple<NDIMS,index_type> idx;

		int height=HeightOfTree(shift);

		Real w=static_cast<Real>(1UL<<(height));

		Real w2=static_cast<Real>(1UL<<(D_FP_POS));

		x*=w;

		nTuple<NDIMS, index_type> h =
		{
			static_cast<index_type>((shift >> (INDEX_DIGITS * 2)) & INDEX_MASK) ,

			static_cast<index_type>((shift >> (INDEX_DIGITS)) & INDEX_MASK),

			static_cast<index_type>(shift & INDEX_MASK)

		};

		for (int i = 0; i < NDIMS; ++i)
		{

			x[i]=x[i]*R_INV_DX[i] - static_cast<Real>(h[i])*w/w2; // [0,1) -> [0,N) N is number of grid

			Real I;

			x[i]=std::modf(x[i],&I);

			if(global_count_[i]<=1) x[i]=0;

			idx[i]=((static_cast<index_type>(I)) <<(D_FP_POS-height)) + h[i];

			auto s=(global_start_[i]<<D_FP_POS);
			auto l=(global_count_[i]<<D_FP_POS);
			idx[i]=(idx[i]-s+l)%l+s;

		}

		return std::move(idx);
	}

	inline index_type coordinates_global_to_local(coordinate_type *px, index_type shift = 0UL) const
	{
		auto idx= (coordinates_to_index(px, shift));

		return ((static_cast<index_type>(idx[0] + COMPACT_INDEX_ZERO) & INDEX_MASK) << (INDEX_DIGITS * 2)) |

		((static_cast<index_type>(idx[1] + COMPACT_INDEX_ZERO) & INDEX_MASK) << (INDEX_DIGITS)) |

		((static_cast<index_type>(idx[2] + COMPACT_INDEX_ZERO) & INDEX_MASK)) |

		shift;

	}

	static Real volume(index_type s)
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
//		return volume_[ele_suffix(s)][HeightOfTree(s)];

		return 1.0;
	}

	static Real inv_volume(index_type s)
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
//		return inv_volume_[ele_suffix(s)][HeightOfTree(s)];
		return 1.0;
	}

//	static Real volume(index_type s)
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
//		return volume_[ele_suffix(s)][HeightOfTree(s)];
//	}
//
//	static Real inv_volume(index_type s)
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
//		return inv_volume_[ele_suffix(s)][HeightOfTree(s)];
//	}

	static Real inv_dual_volume(index_type s)
	{
		return inv_volume(dual(s));
	}
	static Real dual_volume(index_type s)
	{
		return volume(dual(s));
	}
	//***************************************************************************************************

};
// class OcForest

}
// namespace simpla

namespace std
{
template<typename TI> struct iterator_traits;

template<>
struct iterator_traits<simpla::OcForest::iterator>
{
typedef typename simpla::OcForest::iterator iterator;
typedef typename iterator::iterator_category iterator_category;
typedef typename iterator::value_type value_type;
typedef typename iterator::difference_type difference_type;
typedef typename iterator::pointer pointer;
typedef typename iterator::reference reference;

};
}  // namespace std

#endif /* OCTREE_FOREST_H_ */
