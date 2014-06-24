/*
 * uniform_array.h
 *
 *  Created on: 2014年2月21日
 *      Author: salmon
 */

#ifndef UNIFORM_ARRAY_H_
#define UNIFORM_ARRAY_H_

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <limits>
#include <thread>
#include <iterator>
#include "../utilities/ntuple.h"
#include "../utilities/primitives.h"
#include "../utilities/sp_type_traits.h"
#include "../utilities/pretty_stream.h"
#include "../utilities/memory_pool.h"
#include "../parallel/distributed_array.h"
#include "../physics/constants.h"

namespace simpla
{

struct UniformArray
{

	typedef UniformArray this_type;
	static constexpr int MAX_NUM_NEIGHBOUR_ELEMENT = 12;
	static constexpr int MAX_NUM_VERTEX_PER_CEL = 8;
	static constexpr int NDIMS = 3;

	typedef long index_type;

	typedef unsigned long compact_index_type;

	struct iterator;

	struct range;

	typedef range range_type;

	typedef nTuple<NDIMS, Real> coordinates_type;

	typedef std::map<iterator, nTuple<3, coordinates_type>> surface_type;

	//***************************************************************************************************

	UniformArray()
			: depth_of_trees_(0)
	{
	}

	template<typename TDict>
	UniformArray(TDict const & dict)
			: depth_of_trees_(0)
	{
		Load(dict);
	}

	virtual ~UniformArray()
	{
	}

	this_type & operator=(const this_type&) = delete;
	UniformArray(const this_type&) = delete;

	void swap(UniformArray & rhs)
	{
		//FIXME NOT COMPLETE!!
	}

	template<typename TV> using Container=std::shared_ptr<TV>;

	template<int iform, typename TV> inline std::shared_ptr<TV> MakeContainer() const
	{
		return (MEMPOOL.allocate_shared_ptr<TV> (GetLocalNumOfElements(iform)));
	}

	template<typename TDict, typename ...Others>
	void Load(TDict const & dict, Others const& ...)
	{
		try
		{
			LOGGER << "Load UniformArray ";
			SetDimensions(dict["Dimensions"].template as<nTuple<3, index_type>>());
		}
		catch(...)
		{
			PARSER_ERROR("Configure UniformArray error!");
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

	nTuple<NDIMS, index_type> global_start_, global_count_;

	nTuple<NDIMS, index_type> local_outer_start_, local_outer_count_;

	nTuple<NDIMS, index_type> local_inner_start_, local_inner_count_;

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

	void SetDimensions( )
	{
	}

	template<typename TI>
	void SetDimensions(TI const &d)
	{
		for (int i = 0; i < NDIMS; ++i)
		{
			index_type length = d[i] > 0 ? d[i] : 1;

			ASSERT(length<COMPACT_INDEX_ZERO );

			global_start_[i] = 0;
			global_count_[i] = length;
		}

		global_array_.global_start_= global_start_;
		global_array_.global_count_= global_count_;

		UpdateVolume();

		Decompose(1,0,0);

		depth_of_trees_= (count_bits(std::max(global_count_[0],std::max(global_count_[1],global_count_[2]))))+1;

		CHECK(depth_of_trees_);

	}

	nTuple<NDIMS, Real> GetExtents() const
	{

		nTuple<NDIMS, Real> res;

		for (int i = 0; i < NDIMS; ++i )
		{
			res[i]=global_count_[i ]>1?static_cast<Real>(global_count_[i ]):0.0;
		}

		return res;
	}
	coordinates_type GetDx() const
	{
		coordinates_type res;

		for (int i = 0; i < NDIMS; ++i) res[i] = 1.0/static_cast<Real>(global_count_[i]);

		return std::move(res);
	}
	nTuple<NDIMS, index_type> const& GetDimensions() const
	{
		return global_count_;
	}

	index_type GetNumOfElements(int IFORM = VERTEX) const
	{
		return global_count_[0] * global_count_[1] * global_count_[2] * ((IFORM == VERTEX || IFORM == VOLUME) ? 1 : 3);
	}

	nTuple<NDIMS, index_type> const& GetLocalDimensions() const
	{
		return local_outer_count_;
	}
	index_type GetLocalNumOfElements(int IFORM = VERTEX) const
	{
		return local_outer_count_[0] * local_outer_count_[1] * local_outer_count_[2]
		* ((IFORM == VERTEX || IFORM == VOLUME) ? 1 : 3);
	}
	index_type GetLocalMemorySize(int IFORM = VERTEX,int ele_size=1) const
	{
		return local_outer_count_[0] * local_outer_count_[1] * local_outer_count_[2]
		* ((IFORM == VERTEX || IFORM == VOLUME) ? 1 : 3)*ele_size;
	}

	int GetDataSetShape(int IFORM, size_t * global_start = nullptr, size_t * global_count = nullptr, size_t * local_outer_start = nullptr,
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

	//***************************************************************************************************
	//
	// Index Dependent
	//
	//***************************************************************************************************

	//!< signed long is 63bit, unsigned long is 64 bit, add a sign bit
	static constexpr compact_index_type FULL_DIGITS = std::numeric_limits<compact_index_type>::digits;
	static constexpr compact_index_type INDEX_DIGITS = (FULL_DIGITS - CountBits<FULL_DIGITS>::n) / 3;
	static constexpr compact_index_type MAX_INDEX = (1UL << (INDEX_DIGITS-1));
	static constexpr compact_index_type INDEX_MASK = (MAX_INDEX-1);

	const Real dh = 1.0e-18;
	const Real inv_dh = 1.0e18;

	static constexpr compact_index_type _DA=(1UL<<(INDEX_DIGITS*3))|(1UL<<(INDEX_DIGITS*2 ))|(1UL<<(INDEX_DIGITS) );
	static constexpr compact_index_type _MASK=(1UL<<INDEX_DIGITS)-1;
	unsigned int depth_of_trees_;

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

	//mask of direction
	static compact_index_type Compact(nTuple<NDIMS, index_type> const & idx, compact_index_type shift = 0UL)
	{
		return

		( static_cast<compact_index_type>( idx[0] & INDEX_MASK) << (INDEX_DIGITS * 2)) |

		( static_cast<compact_index_type>( idx[1] & INDEX_MASK) << (INDEX_DIGITS)) |

		( static_cast<compact_index_type>( idx[2] & INDEX_MASK)) |

		shift;
	}
	static nTuple<NDIMS, index_type> Decompact(compact_index_type s)
	{
		return nTuple<NDIMS, index_type>(
		{

			static_cast<index_type>((s >> (INDEX_DIGITS * 2)) & INDEX_MASK) ,

			static_cast<index_type>((s >> (INDEX_DIGITS)) & INDEX_MASK) ,

			static_cast<index_type>(s & INDEX_MASK)

		});
	}

	//***************************************************************************************************
	// Coordinates

	/***
	 *
	 * @param s
	 * @return Coordinates range in [0,1)
	 */
	inline nTuple<NDIMS,index_type> CoordinatesToIndex(coordinates_type x) const
	{

		return std::move(nTuple<NDIMS,index_type> (
				{
					static_cast<index_type>(x[0]*inv_dh),
					static_cast<index_type>(x[1]*inv_dh),
					static_cast<index_type>(x[2]*inv_dh),

				}));
	}
	inline coordinates_type GetCoordinates(compact_index_type s)const
	{

		auto d = Decompact(s);

		return coordinates_type(
		{
			static_cast<Real>(d[0])*dh ,
			static_cast<Real>(d[1])*dh ,
			static_cast<Real>(d[2])*dh

		});
	}

	inline coordinates_type CoordinatesLocalToGlobal(compact_index_type s, coordinates_type r)const
	{
		coordinates_type res= GetCoordinates(s)+r*static_cast<Real>(1UL << (INDEX_DIGITS - DepthOfTree(s)));

		return std::move(res);
	}

	inline nTuple<NDIMS,index_type> CoordinatesToIndex (coordinates_type px, compact_index_type shift )const
	{
		return CoordinatesToIndex(&px,shift);
	}

	inline nTuple<NDIMS,index_type> CoordinatesToIndex (coordinates_type *px, compact_index_type shift = 0UL)const
	{

		auto & x = *px;

		x*=inv_dh;

		unsigned int depth = DepthOfTree(shift);

		depth=(depth==0UL)?depth_of_trees_:depth;

		nTuple<NDIMS,index_type> res;

		res = x;

		res+=Decompact(Dual(shift));

		res[0]&= ( ~((1UL << (INDEX_DIGITS -depth))-1));
		res[1]&= ( ~((1UL << (INDEX_DIGITS -depth))-1));
		res[2]&= ( ~((1UL << (INDEX_DIGITS -depth))-1));

		Real w = 1.0/static_cast<Real>(1UL << (INDEX_DIGITS -depth));

		x[0]=(x[0]-static_cast<Real>(res[0]))*w;
		x[1]=(x[1]-static_cast<Real>(res[1]))*w;
		x[2]=(x[2]-static_cast<Real>(res[2]))*w;

		return std::move(res);
	}

	inline compact_index_type CoordinatesGlobalToLocal(coordinates_type *px, compact_index_type shift = 0UL) const
	{
		return Compact(CoordinatesToIndex(px,shift==0?depth_of_trees_:shift));
	}

	Real volume_[8] =
	{	1, // 000
		1,//001
		1,//010
		1,//011
		1,//100
		1,//101
		1,//110
		1//111
	};
	Real inv_volume_[8] =
	{	1, 1, 1, 1, 1, 1, 1, 1};

	Real dual_volume_[8] =
	{	1, 1, 1, 1, 1, 1, 1, 1};

	Real inv_dual_volume_[8] =
	{	1, 1, 1, 1, 1, 1, 1, 1};

	void UpdateVolume()
	{

		for (int i = 0; i < NDIMS; ++i)
		{

			if (global_count_[i]<=1)
			{

				volume_[1UL << i] = 1.0;

				dual_volume_[7 - (1UL << i)] = 1.0;

				inv_volume_[1UL << i] = 1.0;

				inv_dual_volume_[7 - (1UL << i)] = 1.0;

			}
			else
			{

				volume_[1UL << i] = 1.0/static_cast<Real>(global_count_[i]);

				dual_volume_[7 - (1UL << i)] = 1.0/static_cast<Real>(global_count_[i]);

				inv_volume_[1UL << i] = static_cast<Real>(global_count_[i]);

				inv_dual_volume_[7 - (1UL << i)] = static_cast<Real>(global_count_[i]);

			}
		}

		/**
		 *
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
		 *
		 */

		volume_[0] = 1;
//		volume_[1] /* 001 */= dx_[0];
//		volume_[2] /* 010 */= dx_[1];
//		volume_[4] /* 100 */= dx_[2];

		volume_[3] /* 011 */= volume_[1] * volume_[2];
		volume_[5] /* 101 */= volume_[4] * volume_[1];
		volume_[6] /* 110 */= volume_[2] * volume_[4];

		volume_[7] /* 111 */= volume_[1] * volume_[2] * volume_[4];

		dual_volume_[7] = 1;
//		dual_volume_[6] /* 001 */= dx_[0];
//		dual_volume_[5] /* 010 */= dx_[1];
//		dual_volume_[3] /* 100 */= dx_[2];

		dual_volume_[4] /* 011 */= dual_volume_[6] * dual_volume_[5];
		dual_volume_[2] /* 101 */= dual_volume_[3] * dual_volume_[6];
		dual_volume_[1] /* 110 */= dual_volume_[5] * dual_volume_[3];

		dual_volume_[0] /* 111 */= dual_volume_[6] * dual_volume_[5] * dual_volume_[3];

		inv_volume_[0] = 1;
//		inv_volume_[1] /* 001 */= inv_dx_[0];
//		inv_volume_[2] /* 010 */= inv_dx_[1];
//		inv_volume_[4] /* 100 */= inv_dx_[2];

		inv_volume_[3] /* 011 */= inv_volume_[1] * inv_volume_[2];
		inv_volume_[5] /* 101 */= inv_volume_[4] * inv_volume_[1];
		inv_volume_[6] /* 110 */= inv_volume_[2] * inv_volume_[4];

		inv_volume_[7] /* 111 */= inv_volume_[1] * inv_volume_[2] * inv_volume_[4];

		inv_dual_volume_[7] = 1;
//		inv_dual_volume_[6] /* 001 */= inv_dx_[0];
//		inv_dual_volume_[5] /* 010 */= inv_dx_[1];
//		inv_dual_volume_[3] /* 100 */= inv_dx_[2];

		inv_dual_volume_[4] /* 011 */= inv_dual_volume_[6] * inv_dual_volume_[5];
		inv_dual_volume_[2] /* 101 */= inv_dual_volume_[3] * inv_dual_volume_[6];
		inv_dual_volume_[1] /* 110 */= inv_dual_volume_[5] * inv_dual_volume_[3];

		inv_dual_volume_[0] /* 111 */= inv_dual_volume_[6] * inv_dual_volume_[5] * inv_dual_volume_[3];

	}
	Real const & Volume(compact_index_type s)const
	{
		return volume_[NodeId(s)];
	}

	Real InvVolume(compact_index_type s)const
	{
		return inv_volume_[NodeId(s)];
	}

	Real InvDualVolume(compact_index_type s)const
	{
		return inv_dual_volume_[NodeId(s)];
	}
	Real DualVolume(compact_index_type s)const
	{
		return dual_volume_[NodeId(s)];
	}

	//***************************************************************************************************
	//* Auxiliary functions
	//***************************************************************************************************

	static compact_index_type Dual(compact_index_type r)
	{

		return (r & (~(_DA >> (DepthOfTree(r) + 1))))
		| ((~(r & (_DA >> (DepthOfTree(r) + 1)))) & (_DA >> (DepthOfTree(r) + 1)));

	}
	static unsigned int GetCellIndex(compact_index_type r)
	{
		compact_index_type mask=(1UL<<(INDEX_DIGITS-DepthOfTree(r)))-1;

		return r&(~(mask|(mask<<INDEX_DIGITS)|(mask<<(INDEX_DIGITS*2))));
	}
	static unsigned int NodeId(compact_index_type r)
	{
		auto s = (r & (_DA >> (DepthOfTree(r) + 1))) >> (INDEX_DIGITS - DepthOfTree(r) - 1);

		return ((s >> (INDEX_DIGITS * 2)) | (s >> (INDEX_DIGITS - 1)) | (s << 2UL)) & (7UL);
	}

	static unsigned int DepthOfTree(compact_index_type r)
	{
		return r >> (INDEX_DIGITS * 3);
	}

	static compact_index_type Roate(compact_index_type r)
	{

		compact_index_type res;

		res = r & (~(_DA >> (DepthOfTree(r) + 1)));

		res |= (r & ((_DA >> (DepthOfTree(r) + 1))))<<INDEX_DIGITS;

		res |= (r & (1UL << (INDEX_DIGITS*3- DepthOfTree(r) )))>>(INDEX_DIGITS*2);

		return std::move(res);

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

		res = r & (~(_DA >> (DepthOfTree(r) + 1)));

		res |= (r & ((_DA >> (DepthOfTree(r) + 1))))>>INDEX_DIGITS;

		res |= (r & (1UL << (INDEX_DIGITS- DepthOfTree(r) )))<<(INDEX_DIGITS*2);

		return std::move(res);

	}

	static compact_index_type DeltaIndex(compact_index_type r)
	{
		return (r & (_DA >> (DepthOfTree(r) + 1)));
	}

	static compact_index_type DI(unsigned int i,compact_index_type r )
	{
		return (1UL << (INDEX_DIGITS * (NDIMS - i) - DepthOfTree(r) - 1));
	}
	static compact_index_type DeltaIndex(unsigned int i,compact_index_type r )
	{
		return DI(i,r)&r;
	}

	/**
	 * Get component number or vector direction
	 * @param s
	 * @return
	 */
	static index_type ComponentNum(compact_index_type r)
	{
		index_type res = 0;
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

	static index_type IForm(compact_index_type r)
	{
		index_type res = 0;
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

	nTuple<NDIMS, index_type> hash_stride_;

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

	inline index_type Hash(compact_index_type s) const
	{
		auto d =( Decompact(s ) >> (INDEX_DIGITS-depth_of_trees_))-local_outer_start_+local_outer_count_;

		index_type res =

		((d[0] )%local_outer_count_[0]) * hash_stride_[0] +

		((d[1] )%local_outer_count_[1]) * hash_stride_[1] +

		((d[2] )%local_outer_count_[2]) * hash_stride_[2];

		switch (NodeId(s))
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

	compact_index_type GetShift(unsigned int nodeid,compact_index_type h=0UL)const
	{
		h =( h==0)?depth_of_trees_:h;

		return

		(((nodeid>>2) & 1UL)<<(INDEX_DIGITS*3-h-1)) |

		(((nodeid>>1) & 1UL)<<(INDEX_DIGITS*2 -h-1)) |

		((nodeid & 1UL)<<( INDEX_DIGITS-h-1))|

		(h<< (INDEX_DIGITS * 3))
		;
	}

	compact_index_type get_first_node_shift(int iform)const
	{
		compact_index_type res;
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

		return GetShift(res);
	}

	inline unsigned int GetVertices( compact_index_type s, compact_index_type *v) const
	{
		unsigned int n=0;
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
				auto di=DeltaIndex(s);
				v[0] = s + di;
				v[1] = s - di;
			}
			n=2;
			break;

			case FACE:
			{
				auto di = DeltaIndex(Roate(Dual(s)));
				auto dj = DeltaIndex(InverseRoate(Dual(s)));

				v[0] = s - di - dj;
				v[1] = s - di - dj;
				v[2] = s + di + dj;
				v[3] = s + di + dj;
				n=4;
			}
			break;
			case VOLUME:
			{
				auto di = DI(0,s);
				auto dj = DI(1,s);
				auto dk = DI(2,s);

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

	template<int I>
	inline int GetAdjacentCells(Int2Type<I>, Int2Type<I>, compact_index_type s, compact_index_type *v) const
	{
		v[0] = s;
		return 1;
	}

	inline int GetAdjacentCells(Int2Type<EDGE>, Int2Type<VERTEX>, compact_index_type s, compact_index_type *v) const
	{
		v[0] = s + DeltaIndex(s);
		v[1] = s - DeltaIndex(s);
		return 2;
	}

	inline int GetAdjacentCells(Int2Type<FACE>, Int2Type<VERTEX>, compact_index_type s, compact_index_type *v) const
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

		auto di = DeltaIndex(Roate(Dual(s)));
		auto dj = DeltaIndex(InverseRoate(Dual(s)));

		v[0] = s - di - dj;
		v[1] = s - di - dj;
		v[2] = s + di + dj;
		v[3] = s + di + dj;

		return 4;
	}

	inline int GetAdjacentCells(Int2Type<VOLUME>, Int2Type<VERTEX>, compact_index_type s, compact_index_type *v) const
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
		auto di = DI(0,s);
		auto dj = DI(1,s);
		auto dk = DI(2,s);

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

	inline int GetAdjacentCells(Int2Type<VERTEX>, Int2Type<EDGE>, compact_index_type s, compact_index_type *v) const
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

		auto di = DI(0,s);
		auto dj = DI(1,s);
		auto dk = DI(2,s);

		v[0] = s + di;
		v[1] = s - di;

		v[2] = s + dj;
		v[3] = s - dj;

		v[4] = s + dk;
		v[5] = s - dk;

		return 6;
	}

	inline int GetAdjacentCells(Int2Type<FACE>, Int2Type<EDGE>, compact_index_type s, compact_index_type *v) const
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
		auto d1 = DeltaIndex(Roate(Dual(s)));
		auto d2 = DeltaIndex(InverseRoate(Dual(s)));
		v[0] = s - d1;
		v[1] = s + d1;
		v[2] = s - d2;
		v[3] = s + d2;

		return 4;
	}

	inline int GetAdjacentCells(Int2Type<VOLUME>, Int2Type<EDGE>, compact_index_type s, compact_index_type *v) const
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
		auto di = DI(0,s);
		auto dj = DI(1,s);
		auto dk = DI(2,s);

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

	inline int GetAdjacentCells(Int2Type<VERTEX>, Int2Type<FACE>, compact_index_type s, compact_index_type *v) const
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
		auto di = DI(0,s);
		auto dj = DI(1,s);
		auto dk = DI(2,s);

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

	inline int GetAdjacentCells(Int2Type<EDGE>, Int2Type<FACE>, compact_index_type s, compact_index_type *v) const
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

		auto d1 = DeltaIndex(Roate((s)));
		auto d2 = DeltaIndex(InverseRoate((s)));

		v[0] = s - d1;
		v[1] = s + d1;
		v[2] = s - d2;
		v[3] = s + d2;

		return 4;
	}

	inline int GetAdjacentCells(Int2Type<VOLUME>, Int2Type<FACE>, compact_index_type s, compact_index_type *v) const
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

		auto di = DI(0,s);
		auto dj = DI(1,s);
		auto dk = DI(2,s);

		v[0] = s - di;
		v[1] = s + di;

		v[2] = s - di;
		v[3] = s + dj;

		v[4] = s - dk;
		v[5] = s + dk;

		return 6;
	}

	inline int GetAdjacentCells(Int2Type<VERTEX>, Int2Type<VOLUME>, compact_index_type s, compact_index_type *v) const
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

		auto di = DI(0,s);
		auto dj = DI(1,s);
		auto dk = DI(2,s);

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

	inline int GetAdjacentCells(Int2Type<EDGE>, Int2Type<VOLUME>, compact_index_type s, compact_index_type *v) const
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

		auto d1 = DeltaIndex(Roate((s)));
		auto d2 = DeltaIndex(InverseRoate((s)));

		v[0] = s - d1 - d2;
		v[1] = s + d1 - d2;
		v[2] = s - d1 + d2;
		v[3] = s + d1 + d2;
		return 4;
	}

	inline int GetAdjacentCells(Int2Type<FACE>, Int2Type<VOLUME>, compact_index_type s, compact_index_type *v) const
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

		auto d = DeltaIndex(Dual(s));
		v[0] = s + d;
		v[1] = s - d;

		return 2;
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
		typedef index_type difference_type;

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
				auto D = (1UL << (INDEX_DIGITS - DepthOfTree(self_) ));

				self_ += D;

				if ((self_ & _MASK) >= (end_ & _MASK))
				{
					self_ &= ~_MASK;
					self_ |= start_ & _MASK;
					self_ += D << (INDEX_DIGITS);
				}
				if ((self_ & (_MASK<<INDEX_DIGITS)) >= (end_ & (_MASK<<INDEX_DIGITS)))
				{
					self_ &= ~(_MASK<<INDEX_DIGITS);
					self_ |= start_ & (_MASK<<INDEX_DIGITS);
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
				auto D = (1UL << (INDEX_DIGITS - DepthOfTree(self_) ));

				self_ -= D;

				if ((self_ & _MASK) < (start_ & _MASK))
				{
					self_ &= ~_MASK;
					self_ |= (end_ - D) & _MASK;
					self_ -= D << (INDEX_DIGITS);
				}
				if ((self_ & (_MASK<<INDEX_DIGITS)) < (end_ & (_MASK<<INDEX_DIGITS)))
				{
					self_ &= ~ (_MASK<<INDEX_DIGITS);
					self_ |= (end_ - (D << INDEX_DIGITS)) & (_MASK<<INDEX_DIGITS);
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

		static constexpr compact_index_type _MK=_MASK;
		static constexpr compact_index_type _MJ=_MK<<INDEX_DIGITS;
		static constexpr compact_index_type _MI=_MJ<<INDEX_DIGITS;
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
		typedef typename UniformArray::iterator iterator;
		typedef iterator value_type;

		nTuple<NDIMS, index_type> start_;

		nTuple<NDIMS, index_type> count_;

		compact_index_type shift_ = 0UL;

		range( nTuple<NDIMS, index_type> const& start,nTuple<NDIMS, index_type> const & count,compact_index_type shift):
		start_(start),count_(count) ,shift_(shift)
		{
		}

		range(range const& r):start_(r.start_),count_(r.count_), shift_(r.shift_)
		{

		}

		range & operator=(range const&r)
		{
			start_=r.start_;
			count_=r.count_;
			shift_=r.shift_;
			return *this;
		}
		range(range && r):start_(r.start_),count_(r.count_), shift_(r.shift_)
		{
		}

		~range()
		{
		}

		iterator begin() const
		{
			return iterator(
			Compact(start_,shift_) ,
			Compact(start_,shift_) ,
			Compact(start_ + count_,shift_) );
		}
		iterator end() const
		{
			iterator res(begin());

			if (count_[0] * count_[1] * count_[2] > 0)
			{
				res = iterator(

				Compact(start_ + count_ - 1,shift_) ,

				Compact(start_, shift_),

				Compact(start_ + count_,shift_)

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

				(Compact(start_ + count_ ,shift_) ) ,

				(Compact(start_,shift_)),

				(Compact(start_ + count_,shift_) )

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
//		range Split(unsigned int num_process, unsigned int process_num, unsigned int ghost_width = 0) const
//		{
//			int n=0;
//			index_type L=0;
//			for (int i = 0; i < NDIMS; ++i)
//			{
//				if(count_[i]>L)
//				{
//					L=count_[i];
//					n=i;
//				}
//			}
//
//			nTuple<NDIMS,index_type> start,count;
//
//			count = count_;
//			start = start_;
//
//			if ((2 * ghost_width * num_process > count_[n] || num_process > count_[n]) )
//			{
//				if( process_num>0) count=0;
//			}
//			else
//			{
//				start[n] += (count_[n] * process_num ) / num_process;
//				count[n]= (count_[n] * (process_num + 1)) / num_process -(count_[n] * process_num ) / num_process;
//			}
//
//			return range( start,count m);
//		}

	};	// class Range

	template<typename T>
	range Select( unsigned int iform, std::pair<T,T> domain)const
	{
		return Select(domain.first,domain.second,get_first_node_shift(iform));
	}

	range Select(unsigned int iform, coordinates_type xmin, coordinates_type xmax)const
	{

		return range( CoordinatesToIndex(xmin) ,CoordinatesToIndex( xmax) , get_first_node_shift(iform));
	}

	range Select( unsigned int iform, nTuple<NDIMS, index_type> start, nTuple<NDIMS, index_type> end)const
	{
		nTuple<NDIMS, index_type> count= end- start;
		auto flag=Clipping( local_inner_start_, local_inner_count_, &start, &count);

		if (!flag)
		{
			start=local_inner_start_;
			count*=0;
		}

		return range( start,count,get_first_node_shift(iform));
	}

	range Select(unsigned int iform)const
	{
		return range( local_inner_start_,local_inner_count_,get_first_node_shift(iform));
	}

	//***************************************************************************************************

};
// class UniformArray

}
// namespace simpla

namespace std
{
template<typename TI> struct iterator_traits;

template<>
struct iterator_traits<simpla::UniformArray::iterator>
{
typedef typename simpla::UniformArray::iterator iterator;
typedef typename iterator::iterator_category iterator_category;
typedef typename iterator::value_type value_type;
typedef typename iterator::difference_type difference_type;
typedef typename iterator::pointer pointer;
typedef typename iterator::reference reference;

};

}  // namespace std

#endif /* UNIFORM_ARRAY_H_ */
