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
	struct index_type;
	typedef unsigned long compact_index_type;
	typedef nTuple<NDIMS, Real> coordinates_type;
	typedef std::map<index_type, nTuple<3, coordinates_type>> surface_type;

	//!< signed long is 63bit, unsigned long is 64 bit, add a sign bit
	static constexpr unsigned int FULL_DIGITS = std::numeric_limits<compact_index_type>::digits;

	static constexpr unsigned int D_FP_POS = 4; //!< default floating-point position

	static constexpr unsigned int INDEX_DIGITS = (FULL_DIGITS - CountBits<D_FP_POS>::n) / 3;

	static constexpr size_type INDEX_MAX = static_cast<size_type>(((1L) << (INDEX_DIGITS)) - 1);

	static constexpr size_type INDEX_HALF_MAX = INDEX_MAX >> 1;

	static constexpr size_type INDEX_MIN = 0;

	static constexpr compact_index_type NULL_INDEX = ~0UL;

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

//
//	nTuple<NDIMS, size_type> global_end_ = { 1, 1, 1 };

	unsigned long clock_ = 0;

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

	struct index_type
	{
		compact_index_type d;

#define DEF_OP(_OP_)                                                                       \
		inline index_type & operator _OP_##=(compact_index_type r)                           \
		{                                                                                  \
			d =  ( (*this) _OP_ r).d;                                                                \
			return *this ;                                                                  \
		}                                                                                  \
		inline index_type &operator _OP_##=(index_type r)                                   \
		{                                                                                  \
			d = ( (*this) _OP_ r).d;                                                                  \
			return *this;                                                                  \
		}                                                                                  \
                                                                                           \
		inline index_type  operator _OP_(compact_index_type const &r) const                 \
		{                                                                                  \
		return 	std::move(index_type({( ((d _OP_ (r & _MI)) & _MI) |                              \
		                     ((d _OP_ (r & _MJ)) & _MJ) |                               \
		                     ((d _OP_ (r & _MK)) & _MK)                                 \
		                        )& (NO_HEAD_FLAG)}));                                         \
		}                                                                                  \
                                                                                           \
		inline index_type operator _OP_(index_type r) const                                \
		{                                                                                  \
			return std::move(this->operator _OP_(r.d));                                               \
		}                                                                                  \

		DEF_OP(+)
		DEF_OP(-)
		DEF_OP(^)
		DEF_OP(&)
		DEF_OP(|)
#undef DEF_OP

		inline index_type operator>>(unsigned int n) const
		{
			return index_type( { d >> n });
		}
		inline index_type operator<<(unsigned int n) const
		{
			return index_type( { d << n });
		}
		bool operator==(index_type const & rhs) const
		{
			return d == rhs.d;
		}
		bool operator!=(index_type const & rhs) const
		{
			return d != rhs.d;
		}
		bool operator<(index_type const &r) const
		{
			return d < r.d;
		}
		bool operator>(index_type const &r) const
		{
			return d < r.d;
		}

		size_t operator[](unsigned int i) const
		{
			return Get(i);
		}
		size_t Get(unsigned int i) const
		{
			return (d >> (INDEX_DIGITS * (NDIMS - i - 1))) & ((1UL << INDEX_DIGITS) - 1);
		}

		void Set(unsigned int i, size_t v)
		{
			d &= (~((1UL << INDEX_DIGITS) - 1)) << (INDEX_DIGITS * (NDIMS - i - 1));
			d += (v & ((1UL << INDEX_DIGITS) - 1)) << (INDEX_DIGITS * (NDIMS - i - 1));
		}

		void Add(unsigned int i, size_t v)
		{
			d += (v & ((1UL << INDEX_DIGITS) - 1)) << (INDEX_DIGITS * (NDIMS - i - 1));
		}

		//***************************************************************************************************
		//* Auxiliary functions
		//***************************************************************************************************
		void SetNodeId(unsigned int n)
		{

		}

		inline index_type CellIndex() const
		{
			compact_index_type m = (1 << (D_FP_POS - HeightOfTree())) - 1;
			return index_type( { d & (~((m << INDEX_DIGITS * 2) | (m << (INDEX_DIGITS)) | m)) });
		}
		index_type Dual() const
		{
			return index_type(
			        { (d & (~(_DA >> (HeightOfTree() + 1))))
			                | ((~(d & (_DA >> (HeightOfTree() + 1)))) & (_DA >> (HeightOfTree() + 1))) });
		}

		unsigned int HeightOfTree() const
		{
			return d >> (INDEX_DIGITS * 3);
		}

		unsigned int NodeId() const
		{
			auto s = (d & (_DA >> (HeightOfTree() + 1))) >> (D_FP_POS - HeightOfTree() - 1);

			return ((s >> (INDEX_DIGITS * 2)) | (s >> (INDEX_DIGITS - 1)) | (s << 2UL)) & (7UL);
		}

		/**
		 *  rotate vector direction  mask
		 *  (1/2,0,0) => (0,1/2,0) or   (1/2,1/2,0) => (0,1/2,1/2)
		 * @param s
		 * @return
		 */
		index_type NextNode() const
		{
			index_type r;

			r.d = d & ~(_DA >> (HeightOfTree() + 1));

			r |= ((d & (_DI >> (HeightOfTree() + 1))) >> INDEX_DIGITS) |

			((d & (_DJ >> (HeightOfTree() + 1))) >> INDEX_DIGITS) |

			((d & (_DK >> (HeightOfTree() + 1))) << (INDEX_DIGITS * 2))

			;
			return r;
		}

		/**
		 *  rotate vector direction  mask
		 *  (1/2,0,0) => (0,0,1/2) or   (1/2,1/2,0) => (1/2,0,1/2)
		 * @param s
		 * @return
		 */
		index_type PreviousNode() const
		{
			index_type r;

			r.d = d & ~(_DA >> (HeightOfTree() + 1));

			r |= ((d & (_DI >> (HeightOfTree() + 1))) >> (INDEX_DIGITS * 2)) |

			((d & (_DJ >> (HeightOfTree() + 1))) << INDEX_DIGITS) |

			((d & (_DK >> (HeightOfTree() + 1))) << INDEX_DIGITS)

			;

			return r;
		}

		//! get the direction of vector(edge) 0=>x 1=>y 2=>z
		compact_index_type DirectionOfVector() const
		{
			compact_index_type s = (d & (_DA >> (HeightOfTree() + 1))) >> (D_FP_POS - HeightOfTree() - 1);

			return ((s >> (INDEX_DIGITS * 2)) | (s >> (INDEX_DIGITS - 1)) | (s << 2UL)) & (7UL);
		}

		index_type DeltaIndex() const
		{
			return index_type( { d & (_DA >> (HeightOfTree() + 1)) });
		}

		index_type DeltaIndex(unsigned int i) const
		{
			return index_type( { 1UL << (INDEX_DIGITS * (NDIMS - i - 1) + D_FP_POS - HeightOfTree() - 1) });
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

	};

	static index_type ShiftH(index_type s, size_type h = 0)
	{
		return index_type( { (s.d >> h) | (h << (INDEX_DIGITS * 3)) });
	}

	struct iterator
	{
/// One of the @link iterator_tags tag types@endlink.
		typedef std::input_iterator_tag iterator_category;

/// The type "pointed to" by the iterator.
		typedef typename simpla::OcForest::index_type value_type;

/// Distance between iterators is represented as this type.
		typedef typename simpla::OcForest::index_type difference_type;

/// This type represents a pointer-to-value_type.
		typedef value_type* pointer;

/// This type represents a reference-to-value_type.
		typedef value_type& reference;

		OcForest const * mesh;
		value_type s_;

		iterator()
				: mesh(nullptr), s_(value_type( { ~0UL }))
		{
		}
		template<typename ...Args> iterator(OcForest const & m, Args const & ... args)
				: mesh(&m), s_(index_type( { args... }))
		{
		}
		template<typename ...Args> iterator(OcForest const * m, Args const & ... args)
				: mesh(m), s_(index_type( { args... }))
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

		value_type const & operator*() const
		{
			return s_;
		}

		value_type const* operator ->() const
		{
			return &s_;
		}

		iterator & operator ++()
		{
			s_ = mesh->Next(s_);
			return *this;
		}

		iterator operator ++(int)
		{
			iterator res(*this);
			++res;
			return std::move(res);
		}

		size_t Hash() const
		{
			return mesh->Hash(s_);
		}
	};

	struct Range
	{
		typedef typename OcForest::iterator iterator;
		typedef typename iterator::value_type value_type;
	private:
		index_type b_, e_;
		OcForest const * mesh;
	public:
		Range()
				: b_( { 0 }), e_( { 0 }), mesh(nullptr)
		{
		}
		Range(iterator b, iterator e)
				: b_( { b->d }), e_( { e->d }), mesh(b.mesh)
		{
		}
		Range(OcForest const *m, index_type b, index_type e)
				: b_(b), e_(e), mesh(m)
		{
		}

		~Range()
		{
		}
		iterator begin() const
		{
			return iterator(mesh, b_);
		}
		iterator end() const
		{
			return iterator(mesh, e_);
		}
		Range Split(int total, int sub) const
		{
			return mesh->Split(b_, e_, total, sub);
		}
	};

//***************************************************************************************************
// Local Data Set
// local_index  + global_start_   = global_index

//***************************************************************************************************
	index_type INDEX_CENTER = { (INDEX_HALF_MAX << (INDEX_DIGITS * 2)) | (INDEX_HALF_MAX << (INDEX_DIGITS))
	        | (INDEX_HALF_MAX) };

	index_type global_index_start_ = INDEX_CENTER, global_index_end_ = INDEX_CENTER;

	index_type local_index_start_ = INDEX_CENTER, local_index_end_ = INDEX_CENTER;

	nTuple<NDIMS, size_type> memory_dims_ = { 0, 0, 0 };

	nTuple<NDIMS, size_type> memory_start_ = { 0, 0, 0 };

	nTuple<NDIMS, size_type> memory_count_ = { 0, 0, 0 };

	nTuple<NDIMS, size_type> memory_stride_ = { 0, 0, 0 };

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
			size_t length = d[i] > 0 ? d[i] : 1;

			ASSERT(length<INDEX_MAX);

			global_index_start_.Set(i, global_index_start_[i] - ((length / 2) << D_FP_POS));
			global_index_end_.Set(i, global_index_start_[i] + (length << D_FP_POS));
		}
		local_index_start_ = global_index_start_;
		local_index_end_ = global_index_end_;
	}

	nTuple<NDIMS, size_type> GetDimensions() const
	{
		return std::move(GetGlobalDimensions());
	}

	nTuple<NDIMS, size_type> GetGlobalDimensions() const
	{
		index_type count = (global_index_end_ - global_index_start_) >> D_FP_POS;
		return nTuple<NDIMS, size_type>( { count[0], count[1], count[2] });
	}

	nTuple<NDIMS, size_type> GetLocalDimensions() const
	{
		index_type count = (local_index_end_ - local_index_start_) >> D_FP_POS;
		return nTuple<NDIMS, size_type>( { count[0], count[1], count[2] });
	}

	nTuple<NDIMS, Real> GetGlobalExtents() const
	{
		auto dims = GetGlobalDimensions();

		return nTuple<NDIMS, Real>( { static_cast<Real>(dims[0]),

		static_cast<Real>(dims[1]),

		static_cast<Real>(dims[2])

		});
	}

	void Decompose(unsigned int n, unsigned int s, unsigned int gw = 2)
	{

		Decompose(

		nTuple<3, size_t>( { n, 1, 1 }),

		nTuple<3, size_t>( { s, 0, 0 }),

		nTuple<3, size_t>( { gw, gw, gw })

		);
	}

	void Decompose(nTuple<NDIMS, size_t> const & num_process, nTuple<NDIMS, size_t> const & process_num,
	        nTuple<NDIMS, size_t> const & ghost_width)
	{

		for (int i = 0; i < NDIMS; ++i)
		{
			memory_count_[i] = (global_index_end_[i] - global_index_start_[i]) >> (D_FP_POS);

			memory_start_[i] = 0;

			if (2 * ghost_width[i] * num_process[i] > memory_count_[i])
			{
				ERROR << "Mesh is too small to decompose! dims[" << i << "]=" << memory_count_[i]

				<< " process[" << i << "]=" << num_process[i] << " ghost_width=" << ghost_width[i];
			}
			else
			{
				auto start = (memory_count_[i] * process_num[i] / num_process[i]);

				auto end = (memory_count_[i] * (process_num[i] + 1) / num_process[i]);

				memory_count_[i] = start - end;

				if (process_num[i] > 0)
				{
					start -= ghost_width[i];
					memory_start_ = ghost_width[i];
				}
				if (process_num[i] < num_process[i] - 1)
				{
					end += ghost_width[i] << D_FP_POS;
				}

				local_index_start_.Set(i, global_index_start_[i] + (start << D_FP_POS));

				local_index_end_.Set(i, global_index_start_[i] + (end << D_FP_POS));

			}
			memory_dims_[i] = (local_index_end_[i] - local_index_start_[i]) >> D_FP_POS;
		}

		if (array_order_ == SLOW_FIRST)
		{
			memory_stride_[2] = 1;
			memory_stride_[1] = memory_dims_[2];
			memory_stride_[0] = memory_dims_[1] * memory_stride_[1];
		}
		else
		{
			memory_stride_[0] = 1;
			memory_stride_[1] = memory_dims_[0];
			memory_stride_[2] = memory_dims_[1] * memory_stride_[1];
		}

	}

	size_type GetNumOfElements(int IFORM = VERTEX) const
	{
		auto dims = GetGlobalDimensions();
		return dims[0] * dims[1] * dims[2] * ((IFORM == VERTEX || IFORM == VOLUME) ? 1 : 3);
	}

	int GetDataSetShape(int IFORM, size_t * global_dims = nullptr, size_t * global_start = nullptr,
	        size_t * local_dims = nullptr, size_t * local_start = nullptr, size_t * local_count = nullptr,
	        size_t * local_stride = nullptr, size_t * local_block = nullptr) const
	{
		int rank = 0;

		for (int i = 0; i < NDIMS; ++i)
		{
			size_type L = (global_index_end_[i] - global_index_start_[i]) >> D_FP_POS;
			if (L > 1)
			{
				if (global_dims != nullptr)
					global_dims[rank] = L;

				if (global_start != nullptr)
					global_start[rank] = (local_index_start_[i] - INDEX_CENTER[i] + (L << D_FP_POS)) >> D_FP_POS;

				if (local_dims != nullptr)
					local_dims[rank] = (local_index_end_[i] - local_index_start_[i]) >> D_FP_POS;

				if (local_start != nullptr)
					local_start[rank] = memory_start_[i];

				if (local_count != nullptr)
					local_count[rank] = memory_count_[i];

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

	compact_index_type Next(compact_index_type s) const
	{
		return Next(index_type( { s })).d;
	}

	index_type Next(index_type s) const
	{

		// FIXME NEED OPTIMIZE!
		auto n = s.NodeId();

		if (n == 0 || n == 4 || n == 3 || n == 7)
		{

			s.Set(2, s[2] + (1UL << s.HeightOfTree()));

			if (s[2] >= (local_index_end_[2]))
			{
				s.Set(2, local_index_start_[2]);
				s.Set(1, s[1] + (1UL << s.HeightOfTree()));
			}
			if (s[1] >= (local_index_end_[1]))
			{
				s.Set(1, local_index_start_[1]);
				s.Set(0, s[0] + (1UL << s.HeightOfTree()));
			}
			if (s[0] >= local_index_end_[0])
			{
				s.d = -1; // the end
			}
		}

		s = s.NextNode();

		return s;

	}

	Range GetRange(int IFORM = VERTEX) const
	{

		index_type b = local_index_start_, e = local_index_end_;

		if (IFORM == EDGE)
		{
			b.Add(0, 1UL >> (b.HeightOfTree()));
			e |= (_DI >> e.HeightOfTree());
		}
		else if (IFORM == FACE)
		{
			b |= ((_DJ | _DK) >> b.HeightOfTree());
			e |= ((_DJ | _DK) >> e.HeightOfTree());
		}
		else if (IFORM == VOLUME)
		{
			b |= ((_DI | _DJ | _DK) >> b.HeightOfTree());
			e |= ((_DI | _DJ | _DK) >> e.HeightOfTree());
		}

		return Range(this, b, e);
	}

	Range Split(index_type b_, index_type e_, int total, int sub) const
	{

		index_type count = e_ - b_;

		e_ = b_ +

		(((count[0]) * (sub + 1) / total) << (INDEX_DIGITS * 2 + D_FP_POS)) |

		(((count[1]) * (sub + 1) / total) << (INDEX_DIGITS + D_FP_POS)) |

		((count[2]) * (sub + 1) / total) << (D_FP_POS);

		b_ = b_ +

		((count[0] * (sub) / total) << (INDEX_DIGITS * 2 + D_FP_POS)) |

		((count[1] * (sub) / total) << (INDEX_DIGITS + D_FP_POS)) |

		((count[2] * (sub) / total) << (D_FP_POS));

		return Range(this, b_, e_);
	}

	inline size_type Hash(compact_index_type d) const
	{
		return Hash(index_type( { d }));
	}

	inline size_type Hash(index_type s) const
	{

		size_type res = ((s[0] - local_index_start_[0]) >> D_FP_POS) * memory_stride_[0] +

		((s[1] - local_index_start_[1]) >> D_FP_POS) * memory_stride_[1] +

		((s[2] - local_index_start_[2]) >> D_FP_POS) * memory_stride_[2];

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

//***************************************************************************************************

	inline index_type GetIndex(nTuple<3, size_t> const & idx)
	{
		index_type res;
		return res;
	}

	inline coordinates_type GetCoordinates(index_type s) const
	{

		return coordinates_type( {

		static_cast<Real>(s[0]),

		static_cast<Real>(s[1]),

		static_cast<Real>(s[2])

		});

	}

	coordinates_type CoordinatesLocalToGlobal(index_type s, coordinates_type r) const
	{
		Real a = static_cast<double>(1UL << (D_FP_POS - s.HeightOfTree()));

		return coordinates_type( {

		static_cast<Real>(s[0]) + r[0] * a,

		static_cast<Real>(s[1]) + r[1] * a,

		static_cast<Real>(s[2]) + r[2] * a

		});
	}

	inline index_type CoordinatesGlobalToLocalDual(coordinates_type *px, index_type shift = index_type( { 0UL })) const
	{
		return CoordinatesGlobalToLocal(px, shift, 0.5);
	}
	inline index_type CoordinatesGlobalToLocal(coordinates_type *px, index_type shift = index_type( { 0UL }),
	        double round = 0.0) const
	{
		auto & x = *px;

		compact_index_type h = shift.HeightOfTree();

		nTuple<NDIMS, long> idx;

		Real w = static_cast<Real>(1UL << h);

		compact_index_type m = (~((1UL << (D_FP_POS - h)) - 1));

		idx[0] = static_cast<long>(std::floor(round + x[0] + static_cast<double>(shift[0]))) & m;

		x[0] = ((x[0] - idx[0]) * w);

		idx[1] = static_cast<long>(std::floor(round + x[1] + static_cast<double>(shift[1]))) & m;

		x[1] = ((x[1] - idx[1]) * w);

		idx[2] = static_cast<long>(std::floor(round + x[2] + static_cast<double>(shift[2]))) & m;

		x[2] = ((x[2] - idx[2]) * w);

		return index_type(
		        { ((((h << (INDEX_DIGITS * 3)) | (idx[0] << (INDEX_DIGITS * 2)) | (idx[1] << (INDEX_DIGITS)) | (idx[2]))
		                | shift.d)) })

		;

	}

	static Real Volume(index_type s)
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

	static Real InvVolume(index_type s)
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

	static Real InvDualVolume(index_type s)
	{
		return InvVolume(s.Dual());
	}
	static Real DualVolume(index_type s)
	{
		return Volume(s.Dual());
	}

	template<int I>
	inline int GetAdjacentCells(Int2Type<I>, Int2Type<I>, index_type s, index_type *v) const
	{
		v[0] = s;
		return 1;
	}

	inline int GetAdjacentCells(Int2Type<EDGE>, Int2Type<VERTEX>, index_type s, index_type *v) const
	{
		v[0] = s + s.DeltaIndex();
		v[1] = s - s.DeltaIndex();
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

		auto di = s.Dual().NextNode().DeltaIndex();
		auto dj = s.Dual().PreviousNode().DeltaIndex();

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
		auto d1 = s.Dual().NextNode().DeltaIndex();
		auto d2 = s.Dual().PreviousNode().DeltaIndex();
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

		auto d1 = s.NextNode().DeltaIndex();
		auto d2 = s.PreviousNode().DeltaIndex();

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

		auto d1 = s.NextNode().DeltaIndex();
		auto d2 = s.PreviousNode().DeltaIndex();

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

		auto d = s.Dual().DeltaIndex();
		v[0] = s + d;
		v[1] = s - d;

		return 2;
	}

}
;

inline unsigned long make_hash(OcForest::iterator s)
{
	return s.Hash();
}

}
// namespace simpla

#endif /* OCTREE_FOREST_H_ */
