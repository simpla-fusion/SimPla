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
	typedef nTuple<NDIMS, Real> coordinates_type;
	struct iterator;
	typedef std::pair<iterator, iterator> range_type;

	//***************************************************************************************************

	UniformArray()
	{
	}

	template<typename TDict>
	UniformArray(TDict const & dict)
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
		return (MEMPOOL.allocate_shared_ptr<TV> (GetLocalMemorySize(iform)));
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
			PARSER_ERROR("Configure UniformArray error!");}
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

	nTuple<NDIMS, index_type> global_begin_, global_end_ ,global_count_;

	nTuple<NDIMS, index_type> local_outer_begin_, local_outer_end_, local_outer_count_;

	nTuple<NDIMS, index_type> local_inner_begin_, local_inner_end_, local_inner_count_;

	compact_index_type global_begin_compact_index_=0UL;

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
	// _begin          _begin          _begin           _end           _end          _end
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

			global_begin_[i] = (1UL<<(INDEX_DIGITS-MAX_DEPTH_OF_TREE-1)) -length/2;
			global_end_[i] =global_begin_[i]+length;

		}
		global_begin_compact_index_=Compact(global_begin_)<<MAX_DEPTH_OF_TREE;

		global_array_.global_begin_= global_begin_;

		global_array_.global_end_= global_end_;

		global_count_=global_end_-global_begin_;

		Update();

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

		local_inner_begin_=global_array_.local_.inner_begin;
		local_inner_end_=global_array_.local_.inner_end;
		local_inner_count_=local_inner_end_-local_inner_begin_;

		local_outer_begin_=global_array_.local_.outer_begin;
		local_outer_end_=global_array_.local_.outer_end;
		local_outer_count_=local_outer_end_-local_outer_begin_;

		UpdateHash();
	}

	auto GetExtents() const
	DECL_RET_TYPE(std::make_tuple(

			nTuple<NDIMS,Real>(
					{	0,0,0}),

			nTuple<NDIMS,Real>(
					{
						global_count_[0]>1?1.0:0.0,
						global_count_[1]>1?1.0:0.0,
						global_count_[2]>1?1.0:0.0,
					})))

	nTuple<NDIMS, index_type> GetDimensions() const
	{
		return std::move(GetGlobalDimensions());
	}

	nTuple<NDIMS, index_type> GetGlobalDimensions() const
	{
		return global_count_;
	}
	index_type GetNumOfElements(int iform = VERTEX) const
	{
		return GetGlobalNumOfElements(iform);
	}

	index_type GetGlobalNumOfElements(int iform = VERTEX) const
	{
		return NProduct(GetGlobalDimensions()) * ((iform == VERTEX || iform == VOLUME) ? 1 : 3);
	}

	nTuple<NDIMS, index_type> GetLocalDimensions() const
	{
		return local_inner_count_;
	}

	index_type GetMemorySize(int iform = VERTEX ) const
	{
		return GetLocalMemorySize(iform);
	}
	/**
	 *
	 * @return tuple <memory shape, begin, count>
	 */
	std::tuple<nTuple<NDIMS, index_type>,nTuple<NDIMS, index_type>,nTuple<NDIMS, index_type>>
	GetLocalMemoryShape() const
	{
		std::tuple<nTuple<NDIMS, index_type>,nTuple<NDIMS, index_type>,nTuple<NDIMS, index_type>> res;

		std::get<0>(res)=local_outer_count_;

		std::get<1>(res)=local_inner_begin_ - local_outer_begin_;

		std::get<2>(res)=local_inner_count_;

		return std::move(res);
	}

	index_type GetLocalNumOfElements(int iform = VERTEX) const
	{
		return NProduct(std::get<2>(GetLocalMemoryShape())) * ((iform == VERTEX || iform == VOLUME) ? 1 : 3);
	}
	index_type GetLocalMemorySize(int iform = VERTEX ) const
	{
		return NProduct(std::get<0>(GetLocalMemoryShape())) * ((iform == VERTEX || iform == VOLUME) ? 1 : 3);
	}

	int GetDataSetShape(int IFORM, size_t * global_begin = nullptr, size_t * global_end = nullptr, size_t * local_outer_begin = nullptr,
	size_t * local_outer_end = nullptr, size_t * local_inner_begin = nullptr, size_t * local_inner_end = nullptr ) const
	{
		int rank = 0;

		for (int i = 0; i < NDIMS; ++i)
		{
			if ( global_end_[i] -global_begin_[i]>1)
			{

				if (global_begin != nullptr)
				global_begin[rank] = global_begin_[i];

				if (global_end != nullptr)
				global_end[rank] = global_end_[i];

				if (local_outer_begin != nullptr)
				local_outer_begin[rank] = local_inner_begin_[i];

				if (local_outer_end != nullptr)
				local_outer_end[rank] = local_outer_end_[i];

				if (local_inner_begin != nullptr)
				local_inner_begin[rank] = local_inner_begin_[i];

				if (local_inner_end != nullptr)
				local_inner_end[rank] = local_inner_end_[i];

				++rank;
			}

		}
		if (IFORM == EDGE || IFORM == FACE)
		{
			if (global_begin != nullptr)
			global_begin[rank] = 0;

			if (global_end != nullptr)
			global_end[rank] = 3;

			if (local_outer_begin != nullptr)
			local_outer_begin[rank] = 0;

			if (local_outer_end != nullptr)
			local_outer_end[rank] = 3;

			if (local_inner_begin != nullptr)
			local_inner_begin[rank] = 0;

			if (local_inner_end != nullptr)
			local_inner_end[rank] = 3;

			++rank;
		}
		return rank;
	}

	coordinates_type GetDx( ) const
	{
		auto d=GetGlobalDimensions();
		coordinates_type res;

		for (int i = 0; i < NDIMS; ++i)
		{
			res[i] = global_count_[i]>1? (1.0/static_cast<Real>(d[i] )):0.0;
		}

		return std::move(res);
	}

	//***************************************************************************************************
	//
	// Index Dependent
	//
	//***************************************************************************************************

	//!< signed long is 63bit, unsigned long is 64 bit, add a sign bit
	static constexpr compact_index_type FULL_DIGITS = std::numeric_limits<compact_index_type>::digits;

	static constexpr compact_index_type INDEX_DIGITS = (FULL_DIGITS - CountBits<FULL_DIGITS>::n) / 3;

	static constexpr compact_index_type INDEX_MASK = (1UL<<INDEX_DIGITS)-1;

	static constexpr compact_index_type MAX_DEPTH_OF_TREE= 5;

	static constexpr compact_index_type _DI= (1UL<<(INDEX_DIGITS*2+MAX_DEPTH_OF_TREE-1));
	static constexpr compact_index_type _DJ= (1UL<<(INDEX_DIGITS+MAX_DEPTH_OF_TREE-1));
	static constexpr compact_index_type _DK= (1UL<<(MAX_DEPTH_OF_TREE-1));
	static constexpr compact_index_type _DA= _DI|_DJ|_DK;

	static constexpr compact_index_type INDEX_ROOT_MASK= ( (1UL<<(INDEX_DIGITS-MAX_DEPTH_OF_TREE ))-1)<<MAX_DEPTH_OF_TREE;

	static constexpr compact_index_type COMPACT_INDEX_ROOT_MASK=
	INDEX_ROOT_MASK|(INDEX_ROOT_MASK<<INDEX_DIGITS)|(INDEX_ROOT_MASK<<INDEX_DIGITS*2);

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
	static compact_index_type CompactCellIndex(nTuple<NDIMS, index_type> const & idx ,compact_index_type shift)
	{
		return Compact( idx<<MAX_DEPTH_OF_TREE)|shift;
	}

	static nTuple<NDIMS, index_type> DecompactCellIndex(compact_index_type s)
	{
		return std::move(Decompact(s)>>(MAX_DEPTH_OF_TREE));
	}

	//mask of direction
//	static compact_index_type Compact(nTuple<NDIMS, index_type> const & idx )
//	{
//		return
//
//		( static_cast<compact_index_type>( idx[0] & INDEX_MASK) << (INDEX_DIGITS * 2)) |
//
//		( static_cast<compact_index_type>( idx[1] & INDEX_MASK) << (INDEX_DIGITS )) |
//
//		( static_cast<compact_index_type>( idx[2] & INDEX_MASK) )
//
//		;
//	}
	template<typename TS>
	static compact_index_type Compact(nTuple<NDIMS, TS> const & x )
	{
		return

		( (static_cast<compact_index_type>( x[0]) & INDEX_MASK) << (INDEX_DIGITS * 2)) |

		( (static_cast<compact_index_type>( x[1]) & INDEX_MASK) << (INDEX_DIGITS )) |

		( (static_cast<compact_index_type>( x[2]) & INDEX_MASK) )

		;
	}

	static nTuple<NDIMS, index_type> Decompact(compact_index_type s)
	{

		return std::move(nTuple<NDIMS, index_type>(
				{
					static_cast<index_type>((s >> (INDEX_DIGITS * 2)) & INDEX_MASK) ,

					static_cast<index_type>((s >> (INDEX_DIGITS )) & INDEX_MASK) ,

					static_cast<index_type>( s & INDEX_MASK)

				}));
	}

	Real volume_[8];
	Real inv_volume_[8];
	Real dual_volume_[8];
	Real inv_dual_volume_[8];

	nTuple<NDIMS,Real> inv_extents_,extents_;

	void Update()
	{

		for (int i = 0; i < NDIMS; ++i)
		{
			Real L=static_cast<Real>(global_count_[i]);
			if(global_count_[i]<=1)
			{
				extents_[i]=0.0;
				inv_extents_[i]=0.0;

			}
			else
			{
				extents_[i]=static_cast<Real>(global_count_[i] <<MAX_DEPTH_OF_TREE);
				inv_extents_[i]=1.0/extents_[i];
			}

			volume_[1UL << (NDIMS - i - 1)] = 1.0 / L;
			dual_volume_[7 - (1UL << (NDIMS - i - 1))] = 1.0 / L;
			inv_volume_[1UL << (NDIMS - i - 1)] = L;
			inv_dual_volume_[7 - (1UL << (NDIMS - i - 1))] = L;

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
#ifndef ENABLE_SUB_TREE_DEPTH
	Real const & Volume(compact_index_type s) const
	{
		return volume_[NodeId(s)];
	}

	Real InvVolume(compact_index_type s) const
	{
		return inv_volume_[NodeId(s)];
	}

	Real InvDualVolume(compact_index_type s) const
	{
		return inv_dual_volume_[NodeId(s)];
	}
	Real DualVolume(compact_index_type s) const
	{
		return dual_volume_[NodeId(s)];
	}

	Real CellVolume(compact_index_type s)const
	{
		return volume_[1] * volume_[2] * volume_[4];
	}
#else
#error UNIMPLEMENT!!
	Real const & Volume(compact_index_type s) const
	{
		return volume_[NodeId(s)];
	}

	Real InvVolume(compact_index_type s) const
	{
		return inv_volume_[NodeId(s)];
	}

	Real InvDualVolume(compact_index_type s) const
	{
		return inv_dual_volume_[NodeId(s)];
	}
	Real DualVolume(compact_index_type s) const
	{
		return dual_volume_[NodeId(s)];
	}

	Real CellVolume(compact_index_type s)const
	{
		return volume_[1] * volume_[2] * volume_[4];
	}
#endif
	//***************************************************************************************************
	// Coordinates

	/***
	 *
	 * @param s
	 * @return Coordinates range in [0,1)
	 */

	inline coordinates_type
	IndexToCoordinates(nTuple<NDIMS, index_type> const&idx )const
	{
		return std::move(coordinates_type(
				{
					static_cast<Real>(idx[0]- (global_begin_[0]<<MAX_DEPTH_OF_TREE)) * inv_extents_[0] ,
					static_cast<Real>(idx[1]- (global_begin_[1]<<MAX_DEPTH_OF_TREE)) * inv_extents_[1] ,
					static_cast<Real>(idx[2]- (global_begin_[2]<<MAX_DEPTH_OF_TREE)) * inv_extents_[2]
				}));
	}

	inline nTuple<NDIMS, index_type>
	CoordinatesToIndex(coordinates_type x ) const
	{
		return std::move(nTuple<NDIMS, index_type>(
				{
					static_cast<index_type>( x[0] * extents_[0]) + (global_begin_[0]<<MAX_DEPTH_OF_TREE),
					static_cast<index_type>( x[1] * extents_[1]) + (global_begin_[1]<<MAX_DEPTH_OF_TREE),
					static_cast<index_type>( x[2] * extents_[2]) + (global_begin_[2]<<MAX_DEPTH_OF_TREE)
				}));
	}

	inline nTuple<NDIMS, index_type> ToCellIndex(nTuple<NDIMS, index_type> idx)const
	{
		idx=idx>>MAX_DEPTH_OF_TREE;

		return std::move(idx);
	}

	inline coordinates_type
	GetCoordinates(compact_index_type s)const
	{
		return std::move(IndexToCoordinates(Decompact(s)));
	}

	inline coordinates_type
	CoordinatesLocalToGlobal(compact_index_type s ,coordinates_type x )const
	{
		return std::move(CoordinatesLocalToGlobal(std::make_tuple(s,x)));
	}

	template<typename TI> inline coordinates_type
	CoordinatesLocalToGlobal(TI const& v)const
	{

#ifndef ENABLE_SUB_TREE_DEPTH
		static constexpr Real CELL_SCALE_R=static_cast<Real>(1UL<<(MAX_DEPTH_OF_TREE ));
		static constexpr Real INV_CELL_SCALE_R=1.0/CELL_SCALE_R;
		coordinates_type r;
		r = std::get<1>(v) * CELL_SCALE_R + Decompact(std::get<0>(v)-global_begin_compact_index_);

#else

		coordinates_type r= std::get<1>(v)

		* static_cast<Real>(1UL<<(MAX_DEPTH_OF_TREE- DepthOfTree(std::get<0>(v))));

		+Decompact(std::get<0>(v-global_begin_compact_index_));
#endif

		r[0]*=inv_extents_[0];
		r[1]*=inv_extents_[1];
		r[2]*=inv_extents_[2];

		return std::move(r);
	}

	/**
	 *
	 * @param x
	 * @param shift
	 * @return x \in [0,1)
	 */
	inline std::tuple<compact_index_type,coordinates_type>
	CoordinatesGlobalToLocal(coordinates_type x, compact_index_type shift = 0UL) const
	{

		x[0] *= extents_[0];
		x[1] *= extents_[1];
		x[2] *= extents_[2];

#ifndef ENABLE_SUB_TREE_DEPTH

		static constexpr Real CELL_SCALE_R=static_cast<Real>(1UL<<(MAX_DEPTH_OF_TREE ));
		static constexpr Real INV_CELL_SCALE_R=1.0/CELL_SCALE_R;

		compact_index_type s=((Compact(x)+((~shift)&_DA)) &COMPACT_INDEX_ROOT_MASK) |shift;

		x-=Decompact(s);

		x*=INV_CELL_SCALE_R;

		s+=global_begin_compact_index_;
		//*********************************************
#else
		compact_index_type depth = DepthOfTree(shift);

		auto m=( (1UL<<(INDEX_DIGITS-MAX_DEPTH_OF_TREE+depth))-1)<<MAX_DEPTH_OF_TREE;

		m=m|(m<<INDEX_DIGITS)|(m<<(INDEX_DIGITS*2));

		auto s= ((Compact(x)+((~shift)&(_DA>>depth))) &m) |shift;

		x-=Decompact(s);

		x/=static_cast<Real>(1UL<<(MAX_DEPTH_OF_TREE-depth));

		s+= global_begin_compact_index_;

//		nTuple<NDIMS, index_type> idx;
//		idx=x;
//
//		idx=((idx+Decompact((~shift)&(_DA>>depth)))&(~((1UL<<(MAX_DEPTH_OF_TREE-depth))-1)))+Decompact(shift);
//
//		x-=idx;
//
//		x/=static_cast<Real>(1UL<<(MAX_DEPTH_OF_TREE-depth));
//
//		idx+=global_begin_<<MAX_DEPTH_OF_TREE;
//
//		auto s= Compact(idx);
#endif
		return std::move(std::make_tuple( s,x));
	}

//***************************************************************************************************
//* Auxiliary functions
//***************************************************************************************************

	static compact_index_type Dual(compact_index_type r)
	{
#ifndef ENABLE_SUB_TREE_DEPTH
		return (r & (~_DA)) | ((~(r & _DA)) & _DA);
#else
		return (r & (~(_DA >> DepthOfTree(r) )))
		| ((~(r & (_DA >> DepthOfTree(r) ))) & (_DA >> DepthOfTree(r) ));
#endif
	}
	static compact_index_type GetCellIndex(compact_index_type r)
	{
//		compact_index_type mask = (1UL << (INDEX_DIGITS - DepthOfTree(r))) - 1;
//
//		return r & (~(mask | (mask << INDEX_DIGITS) | (mask << (INDEX_DIGITS * 2))));
		return r & COMPACT_INDEX_ROOT_MASK;
	}
	static unsigned int NodeId(compact_index_type s)
	{

#ifndef ENABLE_SUB_TREE_DEPTH
		return
		(((s >> (INDEX_DIGITS*2+MAX_DEPTH_OF_TREE -1))& 1UL) << 2) |

		(((s >>(INDEX_DIGITS +MAX_DEPTH_OF_TREE -1 )) & 1UL) << 1) |

		((s >> (MAX_DEPTH_OF_TREE -1)) & 1UL);
#else
		auto h = DepthOfTree(s);

		return

		(((s >> (INDEX_DIGITS*2+MAX_DEPTH_OF_TREE - h -1))& 1UL) << 2) |

		(((s >>(INDEX_DIGITS +MAX_DEPTH_OF_TREE - h -1 )) & 1UL) << 1) |

		((s >> (MAX_DEPTH_OF_TREE - h -1)) & 1UL);
#endif
	}

	compact_index_type GetShift(unsigned int nodeid, compact_index_type h=0UL) const
	{

#ifndef ENABLE_SUB_TREE_DEPTH
		return

		(((nodeid & 4UL) >> 2) << (INDEX_DIGITS*2+MAX_DEPTH_OF_TREE -1)) |

		(((nodeid & 2UL) >> 1) << (INDEX_DIGITS +MAX_DEPTH_OF_TREE -1 )) |

		((nodeid & 1UL) << (MAX_DEPTH_OF_TREE -1));
#else
		return

		(((nodeid & 4UL) >> 2) << (INDEX_DIGITS*2+MAX_DEPTH_OF_TREE - h -1)) |

		(((nodeid & 2UL) >> 1) << (INDEX_DIGITS +MAX_DEPTH_OF_TREE - h -1 )) |

		((nodeid & 1UL) << (MAX_DEPTH_OF_TREE - h -1)) |

		(h << (INDEX_DIGITS * 3));
#endif
	}

	compact_index_type get_first_node_shift(int iform) const
	{
		compact_index_type nid;
		switch (iform)
		{
			case VERTEX:
			nid = 0;
			break;
			case EDGE:
			nid = 4;
			break;
			case FACE:
			nid = 3;
			break;
			case VOLUME:
			nid = 7;
			break;
		}

		return GetShift(nid );
	}

#ifdef ENABLE_SUB_TREE_DEPTH
	static unsigned int DepthOfTree(compact_index_type r)
	{
		return r >> (INDEX_DIGITS * 3);
	}
#endif

	static compact_index_type Roate(compact_index_type r)
	{

#ifndef ENABLE_SUB_TREE_DEPTH

		return (r & (~_DA))

		| ((r & (((_DI|_DJ) ))) >> INDEX_DIGITS)

		| ((r & (((_DK) )))<< (INDEX_DIGITS * 2));

#else
		compact_index_type h = DepthOfTree(r);

		return (r & (~(_DA >> h)))

		| ((r & (((_DI|_DJ) >> h))) >> INDEX_DIGITS)

		| ((r & (((_DK) >> h)))<< (INDEX_DIGITS * 2));
#endif
	}

	/**
	 *  rotate vector direction  mask
	 *  (1/2,0,0) => (0,0,1/2) or   (1/2,1/2,0) => (1/2,0,1/2)
	 * @param s
	 * @return
	 */

	static compact_index_type InverseRoate(compact_index_type r)
	{

#ifndef ENABLE_SUB_TREE_DEPTH

		return
		(r & (~(_DA)))

		| ((r & (((_DK|_DJ)))) << INDEX_DIGITS)

		| ((r & (((_DI)))) >> (INDEX_DIGITS * 2));

#else
		compact_index_type h = DepthOfTree(r);

		return
		(r & (~(_DA >> h)))

		| ((r & (((_DK|_DJ) >> h))) << INDEX_DIGITS)

		| ((r & (((_DI) >> h))) >> (INDEX_DIGITS * 2));
#endif
	}

	static compact_index_type DeltaIndex(compact_index_type r)
	{
#ifndef ENABLE_SUB_TREE_DEPTH
		return (r & _DA);
#else
		return (r & (_DA >> (DepthOfTree(r))));
#endif
	}

	static compact_index_type DI(unsigned int i, compact_index_type r)
	{
#ifndef ENABLE_SUB_TREE_DEPTH
		return (1UL << (INDEX_DIGITS * (NDIMS-i-1)+MAX_DEPTH_OF_TREE - 1));
#else
		return (1UL << (INDEX_DIGITS * (NDIMS-i-1)+MAX_DEPTH_OF_TREE - DepthOfTree(r) - 1));

#endif
	}
	static compact_index_type DeltaIndex(unsigned int i, compact_index_type r)
	{
		return DI(i, r) & r;
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
			case 4:
			case 3:
			res = 0;
			break;
			case 2:
			case 5:
			res = 1;
			break;
			case 1:
			case 6:
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
			hash_stride_[1] = local_outer_count_[2];
			hash_stride_[0] = local_outer_count_[1] * hash_stride_[1];
		}
		else
		{
			hash_stride_[0] = 1;
			hash_stride_[1] = local_outer_count_[0];
			hash_stride_[2] = local_outer_count_[1] * hash_stride_[1];
		}
	}

	static index_type mod_(index_type a,index_type L)
	{
		return (a+L)%L;
	}

	inline index_type Hash(compact_index_type s) const
	{
		//@FIXME  when idx<0, this is wrong
		nTuple<NDIMS,index_type> d =( Decompact(s)>>MAX_DEPTH_OF_TREE)-local_outer_begin_;

		index_type res =

		mod_( d[0], (local_outer_count_[0] )) * hash_stride_[0] +

		mod_( d[1], (local_outer_count_[1] )) * hash_stride_[1] +

		mod_( d[2], (local_outer_count_[2] )) * hash_stride_[2];

		switch (NodeId(s))
		{
			case 4:
			case 3:
			res = ((res << 1) + res);
			break;
			case 2:
			case 5:
			res = ((res << 1) + res) + 1;
			break;
			case 1:
			case 6:
			res = ((res << 1) + res) + 2;
			break;
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
		typedef index_type difference_type;

/// This type represents a pointer-to-value_type.
		typedef value_type* pointer;

/// This type represents a reference-to-value_type.
		typedef value_type& reference;

		nTuple<NDIMS, index_type> self_;

		nTuple<NDIMS, index_type> begin_, end_;

		compact_index_type shift_;

		bool is_fast_first_ = true;
		iterator( ):shift_(0UL)
		{
		}
		iterator(iterator const & r)
		: self_(r.self_), begin_(r.begin_), end_(r.end_), shift_(r.shift_)
		{
		}
		iterator(iterator && r)
		: self_(r.self_), begin_(r.begin_), end_(r.end_), shift_(r.shift_)
		{
		}
		iterator(nTuple<NDIMS, index_type> s, nTuple<NDIMS, index_type> b, nTuple<NDIMS, index_type> e,
		compact_index_type shift = 0UL)
		: self_(s), begin_(b), end_(e), shift_(shift)
		{
		}

		~iterator()
		{
		}

		iterator & operator=(iterator const & r)

		{
			self_=(r.self_);
			begin_=(r.begin_);
			end_=(r.end_);
			shift_=(r.shift_);

			return *this;
		}
		bool operator==(iterator const & rhs) const
		{
			return self_ == rhs.self_ && shift_ == rhs.shift_;
		}

		bool operator!=(iterator const & rhs) const
		{
			return !(this->operator==(rhs));
		}

		value_type operator*() const
		{
			return Compact(self_<<MAX_DEPTH_OF_TREE)|shift_;
		}

		iterator const * operator->() const
		{
			return this;
		}
		iterator * operator->()
		{
			return this;
		}
		void NextCell()
		{

			if (is_fast_first_)
			{
				++self_[NDIMS - 1];

				for (int i = NDIMS - 1; i > 0; --i)
				{
					if (self_[i] >= end_[i])
					{
						self_[i] = begin_[i];
						++self_[i - 1];
					}
				}
			}
			else
			{
				++self_[0];

				for (int i = 0; i < NDIMS - 1; ++i)
				{
					if (self_[i] >= end_[i])
					{
						self_[i] = begin_[i];
						++self_[i + 1];
					}
				}
			}

		}

		void PreviousCell()
		{

			if (is_fast_first_)
			{
				--self_[NDIMS - 1];

				for (int i = NDIMS - 1; i > 0; --i)
				{
					if (self_[i] < begin_[i])
					{
						self_[i] = end_[i] - 1;
						--self_[i - 1];
					}
				}
			}
			else
			{
				++self_[0];

				for (int i = 0; i < NDIMS; ++i)
				{
					if (self_[i] < begin_[i])
					{
						self_[i] = end_[i] - 1;
						--self_[i + 1];
					}
				}
			}

		}

		iterator & operator ++()
		{
			auto n = NodeId(shift_);

			if (n == 0 || n == 1 || n == 6 || n == 7)
			{
				NextCell();
			}

			shift_ = Roate(shift_);
			return *this;
		}
		iterator operator ++(int) const
		{
			iterator res(*this);
			++res;
			return std::move(res);
		}

		iterator & operator --()
		{
			auto n = NodeId(shift_);

			if (n == 0 || n == 4 || n == 3 || n == 7)
			{
				PreviousCell();
			}

			shift_ = InverseRoate(shift_);
			return *this;
		}

		iterator operator --(int) const
		{
			iterator res(*this);
			--res;
			return std::move(res);
		}

	};	// class iterator

	inline static range_type make_range(nTuple<NDIMS, index_type> begin, nTuple<NDIMS, index_type> end,
	compact_index_type shift = 0UL)
	{

		iterator b(begin, begin, end, shift);

		iterator e(end-1, begin, end, shift);

		e.NextCell();

		return std::move(std::make_pair(std::move(b),std::move( e)));

	}

	range_type Select(unsigned int iform, nTuple<NDIMS, index_type> begin, nTuple<NDIMS, index_type> end) const
	{
		auto flag = Clipping(local_inner_begin_, local_inner_end_, &begin, &end);

		if (!flag)
		{
			begin = local_inner_begin_;
			end = begin;
		}

		return std::move(make_range(begin, end, get_first_node_shift(iform)));

	}

	auto Select(unsigned int iform) const
	DECL_RET_TYPE((Select(iform,local_inner_begin_, local_inner_end_ )))

	template<typename T>
	auto Select(unsigned int iform, std::pair<T, T> domain) const
	DECL_RET_TYPE((Select(iform,domain.first,domain.second )))

	range_type Select(unsigned int iform, coordinates_type xmin, coordinates_type xmax) const
	{
		auto b=ToCellIndex(Decompact(std::get<0>(CoordinatesGlobalToLocal( xmin, get_first_node_shift(iform)))));
		auto e=ToCellIndex(Decompact(std::get<0>(CoordinatesGlobalToLocal( xmax, get_first_node_shift(iform)))));
		e+=1;
		return Select(iform,b,e);
	}
//	DECL_RET_TYPE((Select(iform, CoordinatesToIndex(xmin ) ,CoordinatesToIndex( xmax ) )))
	;

//***************************************************************************************************
// Topology

	inline unsigned int GetVertices(compact_index_type s, compact_index_type *v) const
	{
		unsigned int n = 0;
		switch (IForm(s))
		{
			case VERTEX:
			{
				v[0] = s;
			}
			n = 1;
			break;
			case EDGE:
			{
				auto di = DeltaIndex(s);
				v[0] = s + di;
				v[1] = s - di;
			}
			n = 2;
			break;

			case FACE:
			{
				auto di = DeltaIndex(Roate(Dual(s)));
				auto dj = DeltaIndex(InverseRoate(Dual(s)));

				v[0] = s - di - dj;
				v[1] = s - di - dj;
				v[2] = s + di + dj;
				v[3] = s + di + dj;
				n = 4;
			}
			break;
			case VOLUME:
			{
				auto di = DI(0, s);
				auto dj = DI(1, s);
				auto dk = DI(2, s);

				v[0] = ((s - di) - dj) - dk;
				v[1] = ((s - di) - dj) + dk;
				v[2] = ((s - di) + dj) - dk;
				v[3] = ((s - di) + dj) + dk;

				v[4] = ((s + di) - dj) - dk;
				v[5] = ((s + di) - dj) + dk;
				v[6] = ((s + di) + dj) - dk;
				v[7] = ((s + di) + dj) + dk;
				n = 8;
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
		auto di = DI(0, s);
		auto dj = DI(1, s);
		auto dk = DI(2, s);

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

		auto di = DI(0, s);
		auto dj = DI(1, s);
		auto dk = DI(2, s);

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
		auto di = DI(0, s);
		auto dj = DI(1, s);
		auto dk = DI(2, s);

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
		auto di = DI(0, s);
		auto dj = DI(1, s);
		auto dk = DI(2, s);

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

		auto di = DI(0, s);
		auto dj = DI(1, s);
		auto dk = DI(2, s);

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

		auto di = DI(0, s);
		auto dj = DI(1, s);
		auto dk = DI(2, s);

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

}
;
// class UniformArray
UniformArray::range_type Split(UniformArray::range_type const & range, unsigned int num_process,
        unsigned int process_num, unsigned int ghost_width = 0)
{
	typedef UniformArray::index_type index_type;
	static constexpr int NDIMS = UniformArray::NDIMS;

	auto b = begin(range).self_;
	auto e = (--end(range)).self_ + 1;

	auto shift = begin(range).shift_;

	auto count = e - b;

	int n = 0;
	index_type L = 0;
	for (int i = 0; i < NDIMS; ++i)
	{
		if (count[i] > L)
		{
			L = count[i];
			n = i;
		}
	}

	if ((2 * ghost_width * num_process > count[n] || num_process > count[n]))
	{
		if (process_num > 0)
			count = 0;
	}
	else
	{
		e[n] = b[n] + (count[n] * (process_num + 1)) / num_process;
		b[n] += (count[n] * process_num) / num_process;

	}

	return std::move(UniformArray::make_range(b, e, shift));
}

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
