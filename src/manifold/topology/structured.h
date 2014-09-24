/*
 * structured.h
 *
 *  created on: 2014-2-21
 *      Author: salmon
 */

#ifndef STRUCTURED_H_
#define STRUCTURED_H_

#include <algorithm>
#include <cassert>
#include <cmath>

#include <limits>
#include <thread>
#include <iterator>
#include "../../utilities/ntuple.h"
#include "../../utilities/ntuple_noet.h"
#include "../../utilities/primitives.h"
#include "../../utilities/sp_type_traits.h"
#include "../../utilities/pretty_stream.h"
#include "../../utilities/memory_pool.h"
#include "../../parallel/distributed_array.h"
#include "../../physics/constants.h"
#include "../../numeric/geometric_algorithm.h"
namespace simpla
{
/**
 * \ingroup Topology
 */
/**
 *  \brief  structured mesh, n-dimensional array
 */
struct StructuredMesh
{

	typedef StructuredMesh this_type;
	static constexpr unsigned int MAX_NUM_NEIGHBOUR_ELEMENT = 12;
	static constexpr unsigned int MAX_NUM_VERTEX_PER_CEL = 8;
	static constexpr unsigned int NDIMS = 3;
	static constexpr unsigned int DEFAULT_GHOSTS_WIDTH = 3;
	typedef long index_type;
	typedef unsigned long compact_index_type;
	typedef nTuple<NDIMS, Real> coordinates_type;
	struct iterator;
	typedef std::pair<iterator, iterator> range_type;

	//***************************************************************************************************

	StructuredMesh()
	{
	}

	virtual ~StructuredMesh()
	{
	}

	this_type & operator=(const this_type&) = delete;

	StructuredMesh(const this_type&) = delete;

	void swap(StructuredMesh & rhs)
	{
		//@todo NOT COMPLETE!!
	}

	template<typename TDict>
	bool load(TDict const & dict)
	{
		if (!dict["Dimensions"])
		{
			WARNING << ("Configure  error: no 'Dimensions'!");
			return false;
		}

		set_dimensions(dict["Dimensions"].template as<nTuple<3, index_type>>());

		return true;
	}

	template<typename OS>
	OS & print(OS &os) const
	{
		os << " Dimensions =  " << get_dimensions() << "," << std::endl;

		return os;
	}
	static std::string get_type_as_string_static()
	{
		return "UniformArray";
	}

	std::string get_type_as_string() const
	{
		return get_type_as_string_static();
	}
	std::string save(std::string const &path) const
	{
		return path;
	}
private:
	bool is_ready_ = false;
public:

	bool is_ready() const
	{
		return is_ready_;
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

	bool is_valid() const
	{
		bool res = true;
		for (int i = 0; i < NDIMS; ++i)
		{
			res = res && (global_count_[i] <= 1);
		}
		return !res;
	}

	//! @name Local Data Set
	//! @{

	nTuple<NDIMS, size_t> global_count_;

	nTuple<NDIMS, index_type> global_begin_, global_end_;

	nTuple<NDIMS, index_type> local_outer_begin_, local_outer_end_, local_outer_count_;

	nTuple<NDIMS, index_type> local_inner_begin_, local_inner_end_, local_inner_count_;

	compact_index_type global_begin_compact_index_ = 0UL;

	DistributedArray global_array_;

	//  \verbatim
	//
	//   |----------------|----------------|---------------|--------------|------------|
	//   ^                ^                ^               ^              ^            ^
	//   |                |                |               |              |            |
	//global          local_outer      local_inner    local_inner    local_outer     global
	// _begin          _begin          _begin           _end           _end          _end
	//
	//  \endverbatim

	template<typename TI>
	void set_dimensions(TI const &d)
	{

		for (int i = 0; i < NDIMS; ++i)
		{
			index_type length = d[i] > 0 ? d[i] : 1;

			global_count_[i] = length;
			global_begin_[i] = (1UL << (INDEX_DIGITS - MAX_DEPTH_OF_TREE - 1)) - length / 2;
			global_end_[i] = global_begin_[i] + length;

		}

		global_begin_compact_index_ = compact(global_begin_) << MAX_DEPTH_OF_TREE;

//		local_inner_begin_ = global_begin_;
//		local_inner_end_ = global_end_;
//		local_inner_count_ = global_count_;
//
//		local_outer_begin_ = global_begin_;
//		local_outer_end_ = global_end_;
//		local_outer_count_ = global_count_;

		global_array_.init(NDIMS, global_begin_, global_end_, DEFAULT_GHOSTS_WIDTH);

		local_inner_begin_ = global_array_.local_.inner_begin;
		local_inner_end_ = global_array_.local_.inner_end;
		local_inner_count_ = local_inner_end_ - local_inner_begin_;

		local_outer_begin_ = global_array_.local_.outer_begin;
		local_outer_end_ = global_array_.local_.outer_end;
		local_outer_count_ = local_outer_end_ - local_outer_begin_;

		update();

	}

	bool check_local_memory_bounds(compact_index_type s) const
	{
		auto idx = decompact(s) >> MAX_DEPTH_OF_TREE;
		return

		idx[0] >= local_outer_begin_[0]

		&& idx[0] < local_outer_end_[0]

		&& idx[1] >= local_outer_begin_[1]

		&& idx[1] < local_outer_end_[1]

		&& idx[2] >= local_outer_begin_[2]

		&& idx[2] < local_outer_end_[2]

		;

	}

	auto get_extents() const
	DECL_RET_TYPE(std::make_tuple(

					nTuple<NDIMS,Real>(
							{	0,0,0}),

					nTuple<NDIMS,Real>(
							{
								global_count_[0]>1?1.0:0.0,
								global_count_[1]>1?1.0:0.0,
								global_count_[2]>1?1.0:0.0,
							})))
	;

	auto get_global_dimensions() const
	DECL_RET_TYPE(global_count_)
	;

	auto get_dimensions() const
	DECL_RET_TYPE(global_count_ )
	;

	index_type get_num_of_elements(int iform = VERTEX) const
	{
		return get_global_num_of_elements(iform);
	}

	index_type get_global_num_of_elements(int iform = VERTEX) const
	{
		return NProduct(get_global_dimensions()) * ((iform == VERTEX || iform == VOLUME) ? 1 : 3);
	}

	nTuple<NDIMS, index_type> get_local_dimensions() const
	{
		return local_inner_count_;
	}

	index_type get_memory_size(int iform = VERTEX) const
	{
		return get_local_memory_size(iform);
	}
	/**
	 *
	 * @return tuple <memory shape, begin, count>
	 */
	std::tuple<nTuple<NDIMS, index_type>, nTuple<NDIMS, index_type>, nTuple<NDIMS, index_type>> get_local_memory_shape() const
	{
		std::tuple<nTuple<NDIMS, index_type>, nTuple<NDIMS, index_type>, nTuple<NDIMS, index_type>> res;

		std::get<0>(res) = local_outer_count_;

		std::get<1>(res) = local_inner_begin_ - local_outer_begin_;

		std::get<2>(res) = local_inner_count_;

		return std::move(res);
	}

	index_type get_local_num_of_elements(int iform = VERTEX) const
	{
		return NProduct(std::get<2>(get_local_memory_shape())) * ((iform == VERTEX || iform == VOLUME) ? 1 : 3);
	}
	index_type get_local_memory_size(int iform = VERTEX) const
	{
		return NProduct(std::get<0>(get_local_memory_shape())) * ((iform == VERTEX || iform == VOLUME) ? 1 : 3);
	}

	int get_dataset_shape(int IFORM, size_t * global_begin = nullptr, size_t * global_end = nullptr,
	        size_t * local_outer_begin = nullptr, size_t * local_outer_end = nullptr, size_t * local_inner_begin =
	                nullptr, size_t * local_inner_end = nullptr) const
	{
		int rank = 0;

		for (int i = 0; i < NDIMS; ++i)
		{
			if (global_end_[i] - global_begin_[i] > 1)
			{

				if (global_begin != nullptr)
					global_begin[rank] = global_begin_[i];

				if (global_end != nullptr)
					global_end[rank] = global_end_[i];

				if (local_outer_begin != nullptr)
					local_outer_begin[rank] = local_outer_begin_[i];

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

	int get_dataset_shape(range_type const& range, size_t * global_begin = nullptr, size_t * global_end = nullptr,
	        size_t * local_outer_begin = nullptr, size_t * local_outer_end = nullptr, size_t * local_inner_begin =
	                nullptr, size_t * local_inner_end = nullptr) const
	{
		unsigned int IFORM = IForm(*std::get<0>(range));
		int rank = 0;

		nTuple<NDIMS, index_type> outer_begin, outer_end, outer_count;

		nTuple<NDIMS, index_type> inner_begin, inner_end, inner_count;

		inner_begin = std::get<0>(range).self_;
		inner_end = (std::get<1>(range)--).self_ + 1;

		outer_begin = inner_begin + (-local_inner_begin_ + local_outer_begin_);

		outer_end = inner_end + (-local_inner_end_ + local_outer_end_);

		for (int i = 0; i < NDIMS; ++i)
		{
			if (global_end_[i] - global_begin_[i] > 1)
			{

				if (global_begin != nullptr)
					global_begin[rank] = global_begin_[i];

				if (global_end != nullptr)
					global_end[rank] = global_end_[i];

				if (local_outer_begin != nullptr)
					local_outer_begin[rank] = outer_begin[i];

				if (local_outer_end != nullptr)
					local_outer_end[rank] = outer_end[i];

				if (local_inner_begin != nullptr)
					local_inner_begin[rank] = inner_begin[i];

				if (local_inner_end != nullptr)
					local_inner_end[rank] = inner_end[i];

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

	coordinates_type get_dx() const
	{
		coordinates_type res;

		for (int i = 0; i < NDIMS; ++i)
		{
			res[i] = global_count_[i] > 1 ? (1.0 / static_cast<Real>(global_count_[i])) : 0.0;
		}

		return std::move(res);
	}

	bool in_range(compact_index_type s) const
	{
		auto idx = decompact(s) >> MAX_DEPTH_OF_TREE;

		return true
		        ||

		        ((global_count_[0] > 1 || idx[0] >= global_begin_[0])
		                && (global_count_[1] > 1 || idx[1] >= global_begin_[1])
		                && (global_count_[2] > 1 || idx[2] >= global_begin_[2]));
	}

	bool in_local_range(compact_index_type s) const
	{
		auto idx = decompact(s) >> MAX_DEPTH_OF_TREE;

		return

		((global_count_[0] > 1 || (idx[0] >= local_inner_begin_[0] && idx[0] < local_inner_end_[0])))

		&& ((global_count_[1] > 1 || (idx[1] >= local_inner_begin_[1] && idx[1] < local_inner_end_[1])))

		&& ((global_count_[2] > 1 || (idx[2] >= local_inner_begin_[2] && idx[2] < local_inner_end_[2])));
	}
	//! @}
	//! @name Index Dependent
	//! @{
	//!  signed long is 63bit, unsigned long is 64 bit, add a sign bit
	static constexpr compact_index_type FULL_DIGITS = std::numeric_limits<compact_index_type>::digits;

	static constexpr compact_index_type INDEX_DIGITS = (FULL_DIGITS - CountBits<FULL_DIGITS>::n) / 3;

	static constexpr compact_index_type INDEX_MASK = (1UL << INDEX_DIGITS) - 1;

	static constexpr compact_index_type MAX_DEPTH_OF_TREE = 5;

	static constexpr compact_index_type _DI = (1UL << (INDEX_DIGITS * 2 + MAX_DEPTH_OF_TREE - 1));
	static constexpr compact_index_type _DJ = (1UL << (INDEX_DIGITS + MAX_DEPTH_OF_TREE - 1));
	static constexpr compact_index_type _DK = (1UL << (MAX_DEPTH_OF_TREE - 1));
	static constexpr compact_index_type _DA = _DI | _DJ | _DK;

	static constexpr compact_index_type INDEX_ROOT_MASK = ((1UL << (INDEX_DIGITS - MAX_DEPTH_OF_TREE)) - 1)
	        << MAX_DEPTH_OF_TREE;

	static constexpr compact_index_type COMPACT_INDEX_ROOT_MASK = INDEX_ROOT_MASK | (INDEX_ROOT_MASK << INDEX_DIGITS)
	        | (INDEX_ROOT_MASK << INDEX_DIGITS * 2);

	static constexpr compact_index_type NO_HEAD_FLAG = ~((~0UL) << (INDEX_DIGITS * 3));

	/**
	 *  !! this is obsolete
	 *  \note
	 *  \verbatim
	 * 	Thanks my wife Dr. CHEN Xiang Lan, for her advice on bitwise operation
	 * 	    H          m  I           m    J           m K
	 *  |--------|--------------|--------------|-------------|
	 *  |11111111|00000000000000|00000000000000|0000000000000| <= _MH
	 *  |00000000|11111111111111|00000000000000|0000000000000| <= _MI
	 *  |00000000|00000000000000|11111111111111|0000000000000| <= _MJ
	 *  |00000000|00000000000000|00000000000000|1111111111111| <= _MK
	 *
	 *                      I/J/K
	 *  | INDEX_DIGITS------------------------>|
	 *  |  Root------------------->| Leaf ---->|
	 *  |11111111111111111111111111|00000000000| <=_MRI
	 *  |00000000000000000000000001|00000000000| <=_DI
	 *  |00000000000000000000000000|11111111111| <=_MTI
	 *  | Page NO.->| Tree Root  ->|
	 *  |00000000000|11111111111111|11111111111| <=_MASK
	 *  \endverbatim
	 */
	static compact_index_type compact_cell_index(nTuple<NDIMS, index_type> const & idx, compact_index_type shift)
	{
		return compact(idx << MAX_DEPTH_OF_TREE) | shift;
	}

	static nTuple<NDIMS, index_type> decompact_cell_index(compact_index_type s)
	{
		return std::move(decompact(s) >> (MAX_DEPTH_OF_TREE));
	}

	//mask of direction
//	static compact_index_type compact(nTuple<NDIMS, index_type> const & idx )
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
	static compact_index_type compact(nTuple<NDIMS, TS> const & x)
	{
		return

		((static_cast<compact_index_type>(x[0]) & INDEX_MASK) << (INDEX_DIGITS * 2)) |

		((static_cast<compact_index_type>(x[1]) & INDEX_MASK) << (INDEX_DIGITS)) |

		((static_cast<compact_index_type>(x[2]) & INDEX_MASK))

		;
	}

	static nTuple<NDIMS, index_type> decompact(compact_index_type s)
	{

		return std::move(nTuple<NDIMS, index_type>( { static_cast<index_type>((s >> (INDEX_DIGITS * 2)) & INDEX_MASK),

		static_cast<index_type>((s >> (INDEX_DIGITS)) & INDEX_MASK),

		static_cast<index_type>(s & INDEX_MASK)

		}));
	}
	//! @}

	//! @name Geometry
	//! @{
	Real volume_[8];
	Real inv_volume_[8];
	Real dual_volume_[8];
	Real inv_dual_volume_[8];

	nTuple<NDIMS, Real> inv_extents_, extents_, dx_, inv_dx_;

	bool update()
	{

		for (int i = 0; i < NDIMS; ++i)
		{
			Real L = static_cast<Real>(global_count_[i]);
			if (global_count_[i] <= 1)
			{
				extents_[i] = 0.0;
				inv_extents_[i] = 0.0;
				dx_[i] = 0.0;
				inv_dx_[i] = 0.0;

			}
			else
			{
				extents_[i] = static_cast<Real>((global_count_[i]) << MAX_DEPTH_OF_TREE);
				inv_extents_[i] = 1.0 / extents_[i];
				inv_dx_[i] = static_cast<Real>(global_count_[i]);
				dx_[i] = 1.0 / inv_dx_[i];

			}

			volume_[1UL << (NDIMS - i - 1)] = 1.0 / L;
			dual_volume_[7 - (1UL << (NDIMS - i - 1))] = 1.0 / L;
			inv_volume_[1UL << (NDIMS - i - 1)] = L;
			inv_dual_volume_[7 - (1UL << (NDIMS - i - 1))] = L;

		}

		/**
		 * \note
		 *  \verbatim
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
		 * \endverbatim
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

		is_ready_ = true;

		return is_ready_;
	}
#ifndef ENABLE_SUB_TREE_DEPTH

	Real const & volume(compact_index_type s) const
	{
		return volume_[node_id(s)];
	}

	Real inv_volume(compact_index_type s) const
	{
		return inv_volume_[node_id(s)];
	}

	Real dual_volume(compact_index_type s) const
	{
		return dual_volume_[node_id(s)];
	}

	Real inv_dual_volume(compact_index_type s) const
	{
		return inv_dual_volume_[node_id(s)];
	}

	Real cell_volume(compact_index_type s) const
	{
		return volume_[1] * volume_[2] * volume_[4];
	}

	Real volume(compact_index_type s, std::integral_constant<bool, false>) const
	{
		return volume(s);
	}

	Real inv_volume(compact_index_type s, std::integral_constant<bool, false>) const
	{
		return inv_volume(s);
	}

	Real inv_dual_volume(compact_index_type s, std::integral_constant<bool, false>) const
	{
		return inv_dual_volume(s);
	}
	Real dual_volume(compact_index_type s, std::integral_constant<bool, false>) const
	{
		return dual_volume(s);
	}

	Real volume(compact_index_type s, std::integral_constant<bool, true>) const
	{
		return in_range(s) ? volume(s) : 0.0;
	}

	Real inv_volume(compact_index_type s, std::integral_constant<bool, true>) const
	{
		return in_range(s) ? inv_volume(s) : 0.0;
	}

	Real dual_volume(compact_index_type s, std::integral_constant<bool, true>) const
	{
		return in_range(s) ? dual_volume(s) : 0.0;
	}
	Real inv_dual_volume(compact_index_type s, std::integral_constant<bool, true>) const
	{
		return in_range(s) ? inv_dual_volume(s) : 0.0;
	}

#else
#error UNIMPLEMENT!!
#endif

	//! @}

	//! @name Coordinates
	//! @{

	/***
	 *
	 * @param s
	 * @return Coordinates range in [0,1)
	 */

	inline coordinates_type index_to_coordinates(nTuple<NDIMS, index_type> const&idx) const
	{
		return std::move(
		        coordinates_type(
		                { static_cast<Real>(idx[0] - (global_begin_[0] << MAX_DEPTH_OF_TREE)) * inv_extents_[0],
		                        static_cast<Real>(idx[1] - (global_begin_[1] << MAX_DEPTH_OF_TREE)) * inv_extents_[1],
		                        static_cast<Real>(idx[2] - (global_begin_[2] << MAX_DEPTH_OF_TREE)) * inv_extents_[2] }));
	}

	inline nTuple<NDIMS, index_type> coordinates_to_index(coordinates_type x) const
	{
		return std::move(
		        nTuple<NDIMS, index_type>(
		                { static_cast<index_type>(x[0] * extents_[0]) + (global_begin_[0] << MAX_DEPTH_OF_TREE),
		                        static_cast<index_type>(x[1] * extents_[1]) + (global_begin_[1] << MAX_DEPTH_OF_TREE),
		                        static_cast<index_type>(x[2] * extents_[2]) + (global_begin_[2] << MAX_DEPTH_OF_TREE) }));
	}

	inline nTuple<NDIMS, index_type> to_cell_index(nTuple<NDIMS, index_type> idx) const
	{
		idx = idx >> MAX_DEPTH_OF_TREE;

		return std::move(idx);
	}

	inline coordinates_type get_coordinates(compact_index_type s) const
	{
		return std::move(index_to_coordinates(decompact(s)));
	}

	inline coordinates_type coordinates_local_to_global(compact_index_type s, coordinates_type r) const
	{
#ifndef ENABLE_SUB_TREE_DEPTH
		Real CELL_SCALE_R = static_cast<Real>(1UL << (MAX_DEPTH_OF_TREE));
		Real INV_CELL_SCALE_R = 1.0 / CELL_SCALE_R;

		coordinates_type x;

		x = r + ((decompact(s) >> MAX_DEPTH_OF_TREE) - global_begin_)
		        + 0.5 * (decompact((s & _DA)) >> (MAX_DEPTH_OF_TREE - 1));

		x[0] *= dx_[0];
		x[1] *= dx_[1];
		x[2] *= dx_[2];

#else

		UNIMPLEMENT;
#endif

		return std::move(x);
	}

	template<typename TI> inline auto coordinates_local_to_global(TI const& idx) const
	DECL_RET_TYPE(coordinates_local_to_global(std::get<0>(idx) ,std::get<1>(idx) ))

	/**
	 *
	 * @param x coordinates \f$ x \in \left[0,1\right)\f$
	 * @param shift
	 * @return s,r  s is the largest grid point not greater than x.
	 *       and  \f$ r \in \left[0,1.0\right) \f$ is the normalize  distance between x and s
	 */
	inline std::tuple<compact_index_type, coordinates_type> coordinates_global_to_local(coordinates_type const & x,
	        compact_index_type shift = 0UL) const
	{

#ifndef ENABLE_SUB_TREE_DEPTH

		nTuple<NDIMS, index_type> I = decompact(shift >> (MAX_DEPTH_OF_TREE - 1));

		coordinates_type r;

		r[0] = x[0] * global_count_[0] + global_begin_[0] - 0.5 * static_cast<Real>(I[0]);
		r[1] = x[1] * global_count_[1] + global_begin_[1] - 0.5 * static_cast<Real>(I[1]);
		r[2] = x[2] * global_count_[2] + global_begin_[2] - 0.5 * static_cast<Real>(I[2]);

		I[0] = static_cast<index_type>(std::floor(r[0]));
		I[1] = static_cast<index_type>(std::floor(r[1]));
		I[2] = static_cast<index_type>(std::floor(r[2]));

		r -= I;

		compact_index_type s = (compact(I) << MAX_DEPTH_OF_TREE) | shift;

#else
		compact_index_type depth = DepthOfTree(shift);

		auto m=( (1UL<<(INDEX_DIGITS-MAX_DEPTH_OF_TREE+depth))-1)<<MAX_DEPTH_OF_TREE;

		m=m|(m<<INDEX_DIGITS)|(m<<(INDEX_DIGITS*2));

		auto s= ((compact(x)+((~shift)&(_DA>>depth))) &m) |shift;

		x-=decompact(s);

		x/=static_cast<Real>(1UL<<(MAX_DEPTH_OF_TREE-depth));

		s+= global_begin_compact_index_;

//		nTuple<NDIMS, index_type> idx;
//		idx=x;
//
//		idx=((idx+decompact((~shift)&(_DA>>depth)))&(~((1UL<<(MAX_DEPTH_OF_TREE-depth))-1)))+decompact(shift);
//
//		x-=idx;
//
//		x/=static_cast<Real>(1UL<<(MAX_DEPTH_OF_TREE-depth));
//
//		idx+=global_begin_<<MAX_DEPTH_OF_TREE;
//
//		auto s= compact(idx);
#endif
		return std::move(std::make_tuple(s, r));
	}

	inline std::tuple<compact_index_type, coordinates_type> coordinates_global_to_local_NGP(
	        std::tuple<compact_index_type, coordinates_type> && z) const
	{
		return std::move(coordinates_global_to_local_NGP(std::get<1>(z), std::get<0>(z)));
	}
	/**
	 *
	 * @param x  coordinates \f$ x \in \left[0,1\right)\f$
	 * @param shift
	 * @return s,r   s is thte conmpact index of nearest grid point
	 *    and  \f$ r \in \left[-0.5,0.5\right) \f$   is the normalize  distance between x and s
	 */
	inline std::tuple<compact_index_type, coordinates_type> coordinates_global_to_local_NGP(coordinates_type const & x,
	        compact_index_type shift = 0UL) const
	{

#ifndef ENABLE_SUB_TREE_DEPTH

		nTuple<NDIMS, index_type> I = decompact(shift >> (MAX_DEPTH_OF_TREE - 1));

		coordinates_type r;

		r[0] = x[0] * global_count_[0] + global_begin_[0] - 0.5 * static_cast<Real>(I[0]);
		r[1] = x[1] * global_count_[1] + global_begin_[1] - 0.5 * static_cast<Real>(I[1]);
		r[2] = x[2] * global_count_[2] + global_begin_[2] - 0.5 * static_cast<Real>(I[2]);

		I[0] = static_cast<index_type>(std::floor(r[0] + 0.5));
		I[1] = static_cast<index_type>(std::floor(r[1] + 0.5));
		I[2] = static_cast<index_type>(std::floor(r[2] + 0.5));

		r -= I;

		compact_index_type s = (compact(I) << MAX_DEPTH_OF_TREE) | shift;

#else
		UNIMPLEMENT
#endif
		return std::move(std::make_tuple(s, r));
	}
	//! @}

	//! @name Index auxiliary functions
	//! @{
	static compact_index_type dual(compact_index_type r)
	{
#ifndef ENABLE_SUB_TREE_DEPTH
		return (r & (~_DA)) | ((~(r & _DA)) & _DA);
#else
		return (r & (~(_DA >> DepthOfTree(r) )))
		| ((~(r & (_DA >> DepthOfTree(r) ))) & (_DA >> DepthOfTree(r) ));
#endif
	}
	static compact_index_type get_cell_index(compact_index_type r)
	{
//		compact_index_type mask = (1UL << (INDEX_DIGITS - DepthOfTree(r))) - 1;
//
//		return r & (~(mask | (mask << INDEX_DIGITS) | (mask << (INDEX_DIGITS * 2))));
		return r & COMPACT_INDEX_ROOT_MASK;
	}
	static unsigned int node_id(compact_index_type s)
	{

#ifndef ENABLE_SUB_TREE_DEPTH
		return (((s >> (INDEX_DIGITS * 2 + MAX_DEPTH_OF_TREE - 1)) & 1UL) << 2) |

		(((s >> (INDEX_DIGITS + MAX_DEPTH_OF_TREE - 1)) & 1UL) << 1) |

		((s >> (MAX_DEPTH_OF_TREE - 1)) & 1UL);
#else
		auto h = DepthOfTree(s);

		return

		(((s >> (INDEX_DIGITS*2+MAX_DEPTH_OF_TREE - h -1))& 1UL) << 2) |

		(((s >>(INDEX_DIGITS +MAX_DEPTH_OF_TREE - h -1 )) & 1UL) << 1) |

		((s >> (MAX_DEPTH_OF_TREE - h -1)) & 1UL);
#endif
	}

	compact_index_type get_shift(unsigned int nodeid, compact_index_type h = 0UL) const
	{

#ifndef ENABLE_SUB_TREE_DEPTH
		return

		(((nodeid & 4UL) >> 2) << (INDEX_DIGITS * 2 + MAX_DEPTH_OF_TREE - 1)) |

		(((nodeid & 2UL) >> 1) << (INDEX_DIGITS + MAX_DEPTH_OF_TREE - 1)) |

		((nodeid & 1UL) << (MAX_DEPTH_OF_TREE - 1));
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

		return get_shift(nid);
	}

	static unsigned int get_num_of_comp_per_cell(int iform)
	{
		unsigned int res;
		switch (iform)
		{
		case VERTEX:
			res = 1;
			break;
		case EDGE:
			res = 3;
			break;
		case FACE:
			res = 3;
			break;
		case VOLUME:
			res = 1;
			break;
		}

		return res;
	}

#ifdef ENABLE_SUB_TREE_DEPTH
	static unsigned int DepthOfTree(compact_index_type r)
	{
		return r >> (INDEX_DIGITS * 3);
	}
#endif

	static compact_index_type roate(compact_index_type r)
	{

#ifndef ENABLE_SUB_TREE_DEPTH

		return (r & (~_DA))

		| ((r & (((_DI | _DJ)))) >> INDEX_DIGITS)

		| ((r & (((_DK)))) << (INDEX_DIGITS * 2));

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

	static compact_index_type inverse_roate(compact_index_type s)
	{

#ifndef ENABLE_SUB_TREE_DEPTH

		return (s & (~(_DA)))

		| ((s & (((_DK | _DJ)))) << INDEX_DIGITS)

		| ((s & (((_DI)))) >> (INDEX_DIGITS * 2));

#else
		compact_index_type h = DepthOfTree(s);

		return
		(s & (~(_DA >> h)))

		| ((s & (((_DK|_DJ) >> h))) << INDEX_DIGITS)

		| ((s & (((_DI) >> h))) >> (INDEX_DIGITS * 2));
#endif
	}

	static compact_index_type delta_index(compact_index_type r)
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
		return (1UL << (INDEX_DIGITS * (NDIMS - i - 1) + MAX_DEPTH_OF_TREE - 1));
#else
		return (1UL << (INDEX_DIGITS * (NDIMS-i-1)+MAX_DEPTH_OF_TREE - DepthOfTree(r) - 1));

#endif
	}
	static compact_index_type delta_index(unsigned int i, compact_index_type r)
	{
		return DI(i, r) & r;
	}

	/**
	 * Get component number or vector direction
	 * @param s
	 * @return
	 */
	static index_type component_number(compact_index_type s)
	{
		index_type res = 0;
		switch (node_id(s))
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
		switch (node_id(r))
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
	//! @}

	/**
	 *   @name iterator
	 *   @{
	 */
	//! iterator
	struct iterator
	{

		typedef iterator this_type;

		typedef std::bidirectional_iterator_tag iterator_category;
		typedef compact_index_type value_type;
		typedef long difference_type;
		typedef compact_index_type * pointer;
		typedef compact_index_type reference;

#ifndef USE_FORTRAN_ORDER_ARRAY
		static constexpr unsigned int ARRAY_ORDER = C_ORDER;
#else
		static constexpr unsigned int ARRAY_ORDER=FOTRAN_ORDER;
#endif
		nTuple<NDIMS, index_type> self_;

		nTuple<NDIMS, index_type> begin_, end_;

		compact_index_type shift_;

		iterator()
				: shift_(0UL)
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

		iterator(compact_index_type s)
				: self_(decompact_cell_index(s)), begin_(decompact_cell_index(s)), end_(decompact_cell_index(s) + 1), shift_(
				        delta_index(s))
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
			self_ = (r.self_);
			begin_ = (r.begin_);
			end_ = (r.end_);
			shift_ = (r.shift_);

			return *this;
		}
		bool operator==(iterator const & rhs) const
		{
			return is_same(rhs);
		}

		bool operator!=(iterator const & rhs) const
		{
			return !(this->operator==(rhs));
		}

		value_type operator*() const
		{
			return get();
		}

		iterator const * operator->() const
		{
			return this;
		}
		iterator * operator->()
		{
			return this;
		}

		void reset(compact_index_type s)
		{
			self_ = decompact(s);

			shift_ = delta_index(s);

			NextCell();
			PreviousCell();
		}

		this_type get_end() const
		{
			iterator e(end_ - 1, begin_, end_, shift_);

			e.NextCell();

			return std::move(e);
		}

		void NextCell()
		{
#ifndef USE_FORTRAN_ORDER_ARRAY
			++self_[NDIMS - 1];

			for (int i = NDIMS - 1; i > 0; --i)
			{
				if (self_[i] >= end_[i])
				{
					self_[i] = begin_[i];
					++self_[i - 1];
				}
			}
#else
			++self_[0];

			for (int i = 0; i < NDIMS - 1; ++i)
			{
				if (self_[i] >= end_[i])
				{
					self_[i] = begin_[i];
					++self_[i + 1];
				}
			}
#endif
		}

		void PreviousCell()
		{
#ifndef USE_FORTRAN_ORDER_ARRAY
			--self_[NDIMS - 1];

			for (int i = NDIMS - 1; i > 0; --i)
			{
				if (self_[i] < begin_[i])
				{
					self_[i] = end_[i] - 1;
					--self_[i - 1];
				}
			}

#else

			++self_[0];

			for (int i = 0; i < NDIMS; ++i)
			{
				if (self_[i] < begin_[i])
				{
					self_[i] = end_[i] - 1;
					--self_[i + 1];
				}
			}

#endif //USE_FORTRAN_ORDER_ARRAY
		}

		iterator & operator ++()
		{
			next();
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
			prev();
			return *this;
		}

		iterator operator --(int) const
		{
			iterator res(*this);
			--res;
			return std::move(res);
		}

		compact_index_type get() const
		{
			return compact(self_ << MAX_DEPTH_OF_TREE) | shift_;
		}
		;
		void next()
		{
			auto n = node_id(shift_);

			if (n == 0 || n == 1 || n == 6 || n == 7)
			{
				NextCell();
			}

			shift_ = roate(shift_);
		}
		void prev()
		{
			auto n = node_id(shift_);

			if (n == 0 || n == 4 || n == 3 || n == 7)
			{
				PreviousCell();
			}

			shift_ = inverse_roate(shift_);
			;
		}
		bool is_same(iterator const& rhs) const
		{
			return self_ == rhs.self_ && shift_ == rhs.shift_;
		}

	};	// class iterator

	inline static range_type make_range(nTuple<NDIMS, index_type> begin, nTuple<NDIMS, index_type> end,
	        compact_index_type shift = 0UL)
	{

		iterator b(begin, begin, end, shift);

		return std::move(std::make_pair(std::move(b), std::move(b.get_end())));

	}
	/** @}*/

	/**
	 *  @name Select
	 *  @{
	 */
private:
	range_type select_rectangle_(int iform, nTuple<NDIMS, index_type> const & ib, nTuple<NDIMS, index_type> const & ie,
	        nTuple<NDIMS, index_type> b, nTuple<NDIMS, index_type> e) const
	{
		auto flag = Clipping(ib, ie, &b, &e);

		if (!flag)
		{
			b = ib;
			e = ib;
		}

		return std::move(this_type::make_range(b, e, get_first_node_shift(iform)));

	}
public:
	range_type select(unsigned int iform) const
	{
		return std::move(this_type::make_range(local_inner_begin_, local_inner_end_, get_first_node_shift(iform)));
	}

	range_type select(range_type range) const
	{
		return std::move(range);
	}

	/**
	 * \fn Select
	 * \brief
	 * @param range
	 * @param b
	 * @param e
	 * @return
	 */
	auto select(range_type range, nTuple<NDIMS, index_type> const & b,
	        nTuple<NDIMS, index_type> const &e) const
	                DECL_RET_TYPE(select_rectangle_( IForm(*std::get<0>(range)) ,std::get<0>(range).self_,std::get<1>(range).self_, b, e))

	/**
	 *
	 */
	auto select(range_type range, coordinates_type const & xmin,
	        coordinates_type const &xmax) const
	        DECL_RET_TYPE(select_rectangle_(
					        IForm(*std::get<0>(range)),	//
					std::get<0>(range).self_, std::get<1>(range).self_,//
					to_cell_index(decompact(std::get<0>(coordinates_global_to_local( xmin, get_first_node_shift(IForm(*std::get<0>(range))))))),
					to_cell_index(decompact(std::get<0>(coordinates_global_to_local( xmax, get_first_node_shift(IForm(*std::get<0>(range)))))))+1
			))

	auto select(range_type range1, range_type range2) const
	DECL_RET_TYPE(select_rectangle_(IForm(*std::get<0>(range1)),
					std::get<0>(range1).self_, std::get<1>(range1).self_,	//
			std::get<0>(range2).self_, std::get<1>(range2).self_//
	))

//	template<typename ...Args>
//	auto select(unsigned int iform, Args &&... args) const
//	DECL_RET_TYPE (this_type::select(this_type::select(iform),std::forward<Args>(args)...))

	range_type select_outer(unsigned int iform) const
	{
		return std::move(this_type::make_range(local_outer_begin_, local_outer_end_, get_first_node_shift(iform)));
	}

	/**
	 * \fn Select
	 * \brief
	 * @param range
	 * @param b
	 * @param e
	 * @return
	 */
	auto select_outer(unsigned int iform, nTuple<NDIMS, index_type> const & b, nTuple<NDIMS, index_type> const &e) const
	DECL_RET_TYPE(select_rectangle_( iform,b,e ,local_outer_begin_, local_outer_end_))

	auto select_inner(unsigned int iform, nTuple<NDIMS, index_type> const & b,
	        nTuple<NDIMS, index_type> const & e) const
	        DECL_RET_TYPE(select_rectangle_( iform,b,e ,local_inner_begin_, local_inner_end_))

	/**  @} */
	/**
	 *  @name Hash
	 *  @{
	 */
	static index_type mod_(index_type a, index_type L)
	{
		return (a + L) % L;
	}

	size_t max_hash_value(range_type range) const
	{
		size_t res = NProduct(local_outer_count_);

		auto iform = IForm(*std::get<0>(range));

		if (iform == EDGE || iform == FACE)
		{
			res *= 3;
		}

		return res;
	}

	std::function<size_t(compact_index_type)> make_hash(range_type range) const
	{
		if (!is_ready())
			RUNTIME_ERROR("Mesh is not defined!!");

		std::function<size_t(compact_index_type)> res;

		nTuple<NDIMS, index_type> stride;

		unsigned int iform = IForm(*std::get<0>(range));

		stride[2] = 1;
		stride[1] = local_outer_count_[2] * stride[2];
		stride[0] = local_outer_count_[1] * stride[1];

		res = [=](compact_index_type s)->size_t
		{
			nTuple<NDIMS,index_type> d =( decompact(s)>>MAX_DEPTH_OF_TREE)-local_outer_begin_;

			index_type res =

			mod_( d[0], (local_outer_count_[0] )) * stride[0] +

			mod_( d[1], (local_outer_count_[1] )) * stride[1] +

			mod_( d[2], (local_outer_count_[2] )) * stride[2];

			switch (node_id(s))
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
		};

		//+++++++++++++++++++++++++
//
//		unsigned int iform=IForm(*std::get<0>(range));
//
//#ifdef USE_FORTRAN_ORDER_ARRAY
//		stride[0] = (iform==EDGE||iform==FACE)?3:1;
//		stride[1] = local_outer_count_[0] * stride[0];
//		stride[2] = local_outer_count_[1] * stride[1];
//#else
//		stride[2] = (iform==EDGE||iform==FACE)?3:1;
//		stride[1] = local_outer_count_[2] * stride[2];
//		stride[0] = local_outer_count_[1] * stride[1];
//#endif
//		res=[=](compact_index_type s)->size_t
//		{
//			nTuple<NDIMS,index_type> d =( decompact(s)>>MAX_DEPTH_OF_TREE)-local_outer_begin_;
//
//			return
//
//			mod_( d[0], (local_outer_count_[0] )) * stride[0] +
//
//			mod_( d[1], (local_outer_count_[1] )) * stride[1] +
//
//			mod_( d[2], (local_outer_count_[2] )) * stride[2] +
//
//			component_number(s)
//
//			;
//
//		};

		return std::move(res);
	}

	auto make_hash(unsigned int iform) const
	DECL_RET_TYPE(make_hash(select(iform)))

	/** @}*/

	/** @name   Topology
	 *  @{
	 */

	inline unsigned int get_vertices(compact_index_type s, compact_index_type *v) const
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
			auto di = delta_index(s);
			v[0] = s + di;
			v[1] = s - di;
		}
			n = 2;
			break;

		case FACE:
		{
			auto di = delta_index(roate(dual(s)));
			auto dj = delta_index(inverse_roate(dual(s)));

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

	template<unsigned int I>
	inline unsigned int get_adjacent_cells(std::integral_constant<unsigned int, I>,
	        std::integral_constant<unsigned int, I>, compact_index_type s, compact_index_type *v) const
	{
		v[0] = s;
		return 1;
	}

	inline unsigned int get_adjacent_cells(std::integral_constant<unsigned int, EDGE>,
	        std::integral_constant<unsigned int, VERTEX>, compact_index_type s, compact_index_type *v) const
	{
		v[0] = s + delta_index(s);
		v[1] = s - delta_index(s);
		return 2;
	}

	inline unsigned int get_adjacent_cells(std::integral_constant<unsigned int, FACE>,
	        std::integral_constant<unsigned int, VERTEX>, compact_index_type s, compact_index_type *v) const
	{
		/**
		 * \verbatim
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
		 * \endverbatim
		 *
		 */

		auto di = delta_index(roate(dual(s)));
		auto dj = delta_index(inverse_roate(dual(s)));

		v[0] = s - di - dj;
		v[1] = s - di - dj;
		v[2] = s + di + dj;
		v[3] = s + di + dj;

		return 4;
	}

	inline unsigned int get_adjacent_cells(std::integral_constant<unsigned int, VOLUME>,
	        std::integral_constant<unsigned int, VERTEX>, compact_index_type s, compact_index_type *v) const
	{
		/**
		 * \verbatim
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
		 *   \endverbatim
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

	inline unsigned int get_adjacent_cells(std::integral_constant<unsigned int, VERTEX>,
	        std::integral_constant<unsigned int, EDGE>, compact_index_type s, compact_index_type *v) const
	{
		/**
		 * \verbatim
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
		 * \endverbatim
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

	inline unsigned int get_adjacent_cells(std::integral_constant<unsigned int, FACE>,
	        std::integral_constant<unsigned int, EDGE>, compact_index_type s, compact_index_type *v) const
	{

		/**
		 *\verbatim
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
		 *\endverbatim
		 */
		auto d1 = delta_index(roate(dual(s)));
		auto d2 = delta_index(inverse_roate(dual(s)));
		v[0] = s - d1;
		v[1] = s + d1;
		v[2] = s - d2;
		v[3] = s + d2;

		return 4;
	}

	inline unsigned int get_adjacent_cells(std::integral_constant<unsigned int, VOLUME>,
	        std::integral_constant<unsigned int, EDGE>, compact_index_type s, compact_index_type *v) const
	{

		/**
		 *\verbatim
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
		 *\endverbatim
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

	inline unsigned int get_adjacent_cells(std::integral_constant<unsigned int, VERTEX>,
	        std::integral_constant<unsigned int, FACE>, compact_index_type s, compact_index_type *v) const
	{
		/**
		 *\verbatim
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
		 *\endverbatim
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

	inline unsigned int get_adjacent_cells(std::integral_constant<unsigned int, EDGE>,
	        std::integral_constant<unsigned int, FACE>, compact_index_type s, compact_index_type *v) const
	{

		/**
		 *\verbatim
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
		 *\endverbatim
		 */

		auto d1 = delta_index(roate((s)));
		auto d2 = delta_index(inverse_roate((s)));

		v[0] = s - d1;
		v[1] = s + d1;
		v[2] = s - d2;
		v[3] = s + d2;

		return 4;
	}

	inline unsigned int get_adjacent_cells(std::integral_constant<unsigned int, VOLUME>,
	        std::integral_constant<unsigned int, FACE>, compact_index_type s, compact_index_type *v) const
	{

		/**
		 *\verbatim
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
		 *\endverbatim
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

	inline unsigned int get_adjacent_cells(std::integral_constant<unsigned int, VERTEX>,
	        std::integral_constant<unsigned int, VOLUME>, compact_index_type s, compact_index_type *v) const
	{
		/**
		 *\verbatim
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
		 *\endverbatim
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

	inline unsigned int get_adjacent_cells(std::integral_constant<unsigned int, EDGE>,
	        std::integral_constant<unsigned int, VOLUME>, compact_index_type s, compact_index_type *v) const
	{

		/**
		 *\verbatim
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
		 *\endverbatim
		 */

		auto d1 = delta_index(roate((s)));
		auto d2 = delta_index(inverse_roate((s)));

		v[0] = s - d1 - d2;
		v[1] = s + d1 - d2;
		v[2] = s - d1 + d2;
		v[3] = s + d1 + d2;
		return 4;
	}

	inline unsigned int get_adjacent_cells(std::integral_constant<unsigned int, FACE>,
	        std::integral_constant<unsigned int, VOLUME>, compact_index_type s, compact_index_type *v) const
	{

		/**
		 *\verbatim
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
		 *\endverbatim
		 */

		auto d = delta_index(dual(s));
		v[0] = s + d;
		v[1] = s - d;

		return 2;
	}
	/** @}*/
}
;
// class UniformArray
inline StructuredMesh::range_type split(StructuredMesh::range_type const & range, unsigned int num_process,
        unsigned int process_num, unsigned int ghost_width = 0)
{
	typedef StructuredMesh::index_type index_type;
	static constexpr unsigned int NDIMS = StructuredMesh::NDIMS;

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

	return std::move(StructuredMesh::make_range(b, e, shift));
}

}
// namespace simpla

//namespace std
//{
//
//typename iterator_traits<simpla::StructuredMesh::iterator>::difference_type inline //
//distance(simpla::StructuredMesh::iterator b, simpla::StructuredMesh::iterator e)
//{
//
//	typename simpla::StructuredMesh::iterator::difference_type res;
//
//	--e;
//
//	res = simpla::NProduct((e).self_ - b.self_ + 1);
//
//	switch (simpla::StructuredMesh::IForm(b.shift_))
//	{
//	case simpla::EDGE:
//	case simpla::FACE:
//		res = res * 3
//		        + (simpla::StructuredMesh::component_number(e.shift_)
//		                - simpla::StructuredMesh::component_number(b.shift_)) + 1;
//		break;
//	}
//
//	return res;
//}
//}  // namespace std

#endif /* STRUCTURED_H_ */
