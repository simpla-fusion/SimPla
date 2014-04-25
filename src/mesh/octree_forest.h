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
#include "../fetl/field_rw_cache.h"
#include "../utilities/type_utilites.h"

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

	static constexpr double idh = static_cast<double>(1UL << (D_FP_POS));
	static constexpr double dh = 1.0 / idh;

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

	nTuple<NDIMS, size_type> dims_ = { 1, 1, 1 };

	nTuple<NDIMS, size_type> strides_ = { 0, 0, 0 };

	nTuple<NDIMS, size_type> carray_digits_;

	compact_index_type _MASK = 0;

	//***************************************************************************************************

	OcForest()
			: _MASK(0)
	{

	}

	template<typename TDict>
	OcForest(TDict const & dict)
			: _MASK(0)
	{
	}

	OcForest(nTuple<3, size_type> const & d)
	{
		SetDimensions(d);
	}

	~OcForest()
	{
	}

	this_type & operator=(const this_type&) = delete;

	OcForest(const this_type&) = delete;

	void swap(OcForest & rhs)
	{
		std::swap(dims_, rhs.dims_);
		std::swap(_MASK, rhs._MASK);
	}

	template<typename TDict>
	void Load(TDict const & dict)
	{
		if (dict["Dimensions"])
		{
			LOGGER << "Load OcForest ";
			SetDimensions(dict["Dimensions"].template as<nTuple<3, size_type>>(),
			        dict["ArrayOrder"].template as<std::string>("C_ORDER") == "C_ORDER");
			Update();
		}

	}

	std::string Dump(std::string const &path, bool is_verbose) const
	{
		std::stringstream os;

		os << "\tDimensions =  " << dims_;

		return os.str();
	}

	void Update()
	{
	}

	template<typename TI>
	void SetDimensions(TI const &d, bool C_ORDER = true)
	{
		carray_digits_[0] = D_FP_POS + 1 + ((d[0] > 0) ? (count_bits(d[0]) - 1) : 0);
		carray_digits_[1] = D_FP_POS + 1 + ((d[1] > 0) ? (count_bits(d[1]) - 1) : 0);
		carray_digits_[2] = D_FP_POS + 1 + ((d[2] > 0) ? (count_bits(d[2]) - 1) : 0);
		dims_[0] = 1UL << (carray_digits_[0] - D_FP_POS - 1);
		dims_[1] = 1UL << (carray_digits_[1] - D_FP_POS - 1);
		dims_[2] = 1UL << (carray_digits_[2] - D_FP_POS - 1);

		_MASK =

		(((1UL << (carray_digits_[0] - 1)) - 1) << (INDEX_DIGITS * 2)) |

		(((1UL << (carray_digits_[1] - 1)) - 1) << (INDEX_DIGITS)) |

		(((1UL << (carray_digits_[2] - 1)) - 1))

		;

		if (C_ORDER)
		{
			strides_[2] = 1;
			strides_[1] = dims_[2];
			strides_[0] = dims_[1] * strides_[1];
		}
		else
		{
			strides_[0] = 1;
			strides_[1] = dims_[0];
			strides_[2] = dims_[1] * strides_[1];
		}

//		CHECK(carray_digits_);
//		CHECK_BIT(_MASK);
//
//		CHECK_BIT(_DI);
//		CHECK_BIT(_MI);
//		CHECK_BIT(_MRI);
//		CHECK_BIT(_MTI);
//		CHECK_BIT(_MASK);
//
//		CHECK_BIT(_DJ);
//		CHECK_BIT(_MJ);
//		CHECK_BIT(_MRJ);
//		CHECK_BIT(_MTJ);
//		CHECK_BIT(_MASK);
//
//		CHECK_BIT(_DK);
//		CHECK_BIT(_MK);
//		CHECK_BIT(_MRK);
//		CHECK_BIT(_MTK);
//		CHECK_BIT(_MASK);

	}

//***************************************************************************************************

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
	};
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
		template<typename ...Args>
		iterator(OcForest const & m, Args const & ... args)
				: mesh(&m), s_(index_type( { args... }))
		{
		}
		template<typename ...Args>
		iterator(OcForest const * m, Args const & ... args)
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
		compact_index_type b_, e_;
		OcForest const * mesh;
	public:
		Range()
				: b_(0), e_(0), mesh(nullptr)
		{
		}
		Range(iterator b, iterator e)
				: b_(b->d), e_(e->d), mesh(b.mesh)
		{
		}
		Range(OcForest const *m, compact_index_type b, compact_index_type e)
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
			compact_index_type b = b_, e = b_;
			nTuple<3, compact_index_type> L;
			L[0] = (((e_ - b_) & _MI) >> (D_FP_POS + INDEX_DIGITS * 2));
			L[1] = (((e_ - b_) & _MJ) >> (D_FP_POS + INDEX_DIGITS));
			L[2] = (((e_ - b_) & _MK) >> (D_FP_POS));

			for (int i = 0; i < 3; ++i)
			{
				if (L[i] == 0)
					L[i] = mesh->dims_[i];
			}
			if (L[0] >= total)
			{
				b += (L[0] * sub / total) << (INDEX_DIGITS * 2 + D_FP_POS);
				e = b + ((L[0] / total) << (INDEX_DIGITS * 2 + D_FP_POS));
			}
			else if (L[1] >= total)
			{
				b += (L[1] * sub / total) << (INDEX_DIGITS + D_FP_POS);
				e = b + ((L[1] / total) << (INDEX_DIGITS + D_FP_POS));
			}
			else if (L[2] >= total)
			{
				b += (L[2] * sub / total) << (D_FP_POS);
				e = b + ((L[2] / total) << (D_FP_POS));
			}
			else
			{
				ERROR << ("TODO: board split");
			}

			e = mesh->CalCarray(e);

			return Range(mesh, b, e);
		}

	};

	Range GetRange(int IFORM = VERTEX) const
	{

		compact_index_type b = 0, e = 0;

		if (IFORM == EDGE)
		{
			b |= (_DI >> 1);
		}
		else if (IFORM == FACE)
		{
			b |= ((_DJ | _DK) >> 1);
		}
		else if (IFORM == VOLUME)
		{
			b |= ((_DI | _DJ | _DK) >> 1);
		}

		e = b | ((dims_[0]) << (INDEX_DIGITS * 2 + D_FP_POS));

		return Range(this, b, e);
	}

	compact_index_type CalCarray(compact_index_type s) const
	{
		auto bit = s & (1UL << (carray_digits_[2] - 1));
		s = (s & (~bit)) + (bit << (INDEX_DIGITS + D_FP_POS - carray_digits_[2] + 1 - H(s)));
		bit = s & (1UL << (carray_digits_[1] + INDEX_DIGITS - 1));
		s = (s & (~bit)) + (bit << (INDEX_DIGITS + D_FP_POS - carray_digits_[1] + 1 - H(s)));
		return s;
	}

	compact_index_type Next(compact_index_type s) const
	{
		auto n = _N(s);

		if (n == 0 || n == 4 || n == 3 || n == 7)
		{
			s = CalCarray(s + (_DK >> H(s)));
//			auto bit = s & (1UL << (carray_digits_[2] - 1));
//			s = (s & (~bit)) + (bit << (INDEX_DIGITS + D_FP_POS - carray_digits_[2] + 1 - H(s)));
//			bit = s & (1UL << (carray_digits_[1] + INDEX_DIGITS - 1));
//			s = (s & (~bit)) + (bit << (INDEX_DIGITS + D_FP_POS - carray_digits_[1] + 1 - H(s)));
		}

		s = _R(s);

		return s;
	}

	index_type Next(index_type s) const
	{
		return index_type( { Next(s.d) });
	}

//***************************************************************************************************

	inline size_type Hash(compact_index_type d) const
	{
		d &= _MASK;

		//FIXME something wrong at here , FIX IT!!!
		return Hash((I(d) >> D_FP_POS), (J(d) >> D_FP_POS), (K(d) >> D_FP_POS), _N(d));

	}

	inline size_type Hash(index_type s) const
	{

		return std::move(Hash(s.d));

	}

	inline size_type Hash(size_type i, size_type j, size_type k) const
	{
		return i * strides_[0] + j * strides_[1] + k * strides_[2];

	}
	inline size_type Hash(size_type i, size_type j, size_type k, size_type m) const
	{

		size_type res = Hash(i, j, k);
		switch (m)
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

	inline compact_index_type GetCellIndex(compact_index_type s) const
	{
		compact_index_type m = (1 << (D_FP_POS - H(s))) - 1;
		return s & (_MASK & ~((m << INDEX_DIGITS * 2) | (m << (INDEX_DIGITS)) | m));
	}

	inline index_type GetCellIndex(index_type s) const
	{
		return index_type( { GetCellIndex(s.d) });
	}

//***************************************************************************************************

	nTuple<NDIMS, size_type> const & GetDimensions() const
	{
		return dims_;
	}

	size_type GetNumOfElements(int IFORM = VERTEX) const
	{
		return dims_[0] * dims_[1] * dims_[2] * ((IFORM == VERTEX || IFORM == VOLUME) ? 1 : 3);
	}

	inline std::vector<size_type> GetShape(int IFORM) const
	{
		std::vector<size_type> res;

		for (int i = 0; i < NDIMS; ++i)
		{
			if (dims_[i] > 1)
				res.push_back(dims_[i]);
		}
		if (IFORM == EDGE || IFORM == FACE)
		{
			res.push_back(3);
		}

		return std::move(res);
	}

	inline coordinates_type GetCoordinates(compact_index_type s) const
	{
		s &= _MASK;

		return coordinates_type( {

		static_cast<Real>(I(s)) * dh,

		static_cast<Real>(J(s)) * dh,

		static_cast<Real>(K(s)) * dh

		});

	}

	inline coordinates_type GetCoordinates(index_type s) const
	{
		return std::move(GetCoordinates(s.d));

	}

	coordinates_type CoordinatesLocalToGlobal(index_type s, coordinates_type r) const
	{
		coordinates_type x = GetCoordinates(s);
		Real a = static_cast<double>(1UL << (D_FP_POS - H(s))) * dh;

		x[0] = (dims_[0] > 1) ? (x[0] + r[0] * a) : 0;

		x[1] = (dims_[1] > 1) ? (x[1] + r[1] * a) : 0;

		x[2] = (dims_[2] > 1) ? (x[2] + r[2] * a) : 0;

		return x;
	}

	inline index_type CoordinatesGlobalToLocalDual(coordinates_type * x, compact_index_type shift = 0UL) const
	{
		return CoordinatesGlobalToLocal(x, shift, 0.5);
	}
	inline index_type CoordinatesGlobalToLocal(coordinates_type * x, compact_index_type shift = 0UL,
	        double round = 0.0) const
	{
		compact_index_type h = H(shift);

		nTuple<NDIMS, long> idx;

		Real w = static_cast<Real>(1UL << h);

		compact_index_type m = (~((1UL << (D_FP_POS - h)) - 1));

		idx[0] = static_cast<long>(std::floor(round + (*x)[0] * idh + static_cast<double>(I(shift)))) & m;

		(*x)[0] = (dims_[0] > 1) ? (((*x)[0] - idx[0] * dh) * w) : 0.0;

		idx[1] = static_cast<long>(std::floor(round + (*x)[1] * idh + static_cast<double>(J(shift)))) & m;

		(*x)[1] = (dims_[0] > 1) ? (((*x)[1] - idx[1] * dh) * w) : 0.0;

		idx[2] = static_cast<long>(std::floor(round + (*x)[2] * idh + static_cast<double>(K(shift)))) & m;

		(*x)[2] = (dims_[0] > 1) ? (((*x)[2] - idx[2] * dh) * w) : 0.0;

		return index_type( { ((((h << (INDEX_DIGITS * 3)) |

		((idx[0] & ((1UL << carray_digits_[0]) - 1)) << (INDEX_DIGITS * 2)) |

		((idx[1] & ((1UL << carray_digits_[1]) - 1)) << (INDEX_DIGITS)) |

		(idx[2] & ((1UL << carray_digits_[2]) - 1))) |

		shift)) & _MASK });

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

		        1, 1.0 / 8, 1.0 / 32, 1.0 / 128   // 111

		};

		return volume_[_N(s)][H(s)];
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

		        1, 8, 32, 128   // 111

		        };

		return inv_volume_[_N(s)][H(s)];
	}

	static Real InvDualVolume(index_type s)
	{
		return InvVolume(_Dual(s));
	}
	static Real DualVolume(index_type s)
	{
		return Volume(_Dual(s));
	}
//***************************************************************************************************
//* Auxiliary functions
//***************************************************************************************************

	static size_type H(compact_index_type s)
	{
		return s >> (INDEX_DIGITS * 3);
	}

	static size_type H(index_type s)
	{
		return H(s.d);
	}

	static compact_index_type ShiftH(compact_index_type s, compact_index_type h = 0)
	{
		return (s >> h) | (h << (INDEX_DIGITS * 3));
	}
	static index_type ShiftH(index_type s, compact_index_type h)
	{
		return index_type( { ShiftH(s.d, h) });
	}

	size_type I(compact_index_type s) const
	{
		return (s & _MI & _MASK) >> (INDEX_DIGITS * 2);
	}

	size_type I(index_type s) const
	{
		return I(s.d);
	}

	size_type J(compact_index_type s) const
	{
		return (s & _MJ & _MASK) >> (INDEX_DIGITS);
	}

	size_type J(index_type s) const
	{
		return J(s.d);
	}
	size_type K(compact_index_type s) const
	{
		return (s & _MK & _MASK);
	}

	size_type K(index_type s) const
	{
		return K(s.d);
	}

	/**
	 *  rotate vector direction  mask
	 *  (1/2,0,0) => (0,1/2,0) or   (1/2,1/2,0) => (0,1/2,1/2)
	 * @param s
	 * @return
	 */
	index_type _R(index_type s) const
	{
		s.d = _R(s.d);
		return s;
	}

	compact_index_type _R(compact_index_type s) const
	{
		compact_index_type r = s;

		r &= ~(_DA >> (H(s) + 1));

		r |= ((s & (_DI >> (H(s) + 1))) >> INDEX_DIGITS) |

		((s & (_DJ >> (H(s) + 1))) >> INDEX_DIGITS) |

		((s & (_DK >> (H(s) + 1))) << (INDEX_DIGITS * 2))

		;
		return r;
	}

	/**
	 *  rotate vector direction  mask
	 *  (1/2,0,0) => (0,0,1/2) or   (1/2,1/2,0) => (1/2,0,1/2)
	 * @param s
	 * @return
	 */
	index_type _RR(index_type s) const
	{
		s.d = _RR(s.d);
		return s;
	}

	compact_index_type _RR(compact_index_type s) const
	{
		compact_index_type r = s;
		r &= ~(_DA >> (H(s) + 1));

		r |= ((s & (_DI >> (H(s) + 1))) >> (INDEX_DIGITS * 2)) |

		((s & (_DJ >> (H(s) + 1))) << INDEX_DIGITS) |

		((s & (_DK >> (H(s) + 1))) << INDEX_DIGITS)

		;
		return r;
	}

	/**
	 *    (1/2,0,1/2) => (0,1/2,0) or   (1/2,0,0) => (0,1/2,1/2)
	 * @param s
	 * @return
	 */
	static index_type _Dual(index_type s)
	{
		s.d = _Dual(s.d);
		return s;
	}

	static compact_index_type _Dual(compact_index_type s)
	{
		return std::move((s & (~(_DA >> (H(s) + 1)))) | ((~(s & (_DA >> (H(s) + 1)))) & (_DA >> (H(s) + 1))));
	}

//! get the direction of vector(edge) 0=>x 1=>y 2=>z
	static size_type _N(compact_index_type s)
	{

		s = (s & (_DA >> (H(s) + 1))) >> (D_FP_POS - H(s) - 1);

		return ((s >> (INDEX_DIGITS * 2)) | (s >> (INDEX_DIGITS - 1)) | (s << 2UL)) & (7UL);
	}
	static size_type _N(index_type s)
	{
		return std::move(_N(s.d));
	}
	/**
	 * Get component number or vector direction
	 * @param s
	 * @return
	 */
	static size_type _C(compact_index_type s)
	{
		size_type res = 0;
		switch (_N(s))
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
	static size_type _C(index_type s)
	{
		return std::move(_C(s.d));
	}
	static index_type _D(index_type s)
	{
		s.d = _D(s.d);
		return s;
	}
	static compact_index_type _D(compact_index_type s)
	{
		return s & (_DA >> (H(s) + 1));

	}

	static int _IForm(compact_index_type s)
	{
		size_type res = 0;
		switch (_N(s))
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
	static int _IForm(index_type s)
	{
		return (_IForm(s.d));
	}
	template<int I>
	inline int GetAdjacentCells(Int2Type<I>, Int2Type<I>, index_type s, index_type *v) const
	{
		v[0] = s;
		return 1;
	}

	inline int GetAdjacentCells(Int2Type<EDGE>, Int2Type<VERTEX>, index_type s, index_type *v) const
	{
		v[0] = s + _D(s);
		v[1] = s - _D(s);
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

		auto di = _D(_R(_Dual(s)));
		auto dj = _D(_RR(_Dual(s)));

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
		auto di = _DI >> (H(s) + 1);
		auto dj = _DJ >> (H(s) + 1);
		auto dk = _DK >> (H(s) + 1);

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

		auto di = _DI >> (H(s) + 1);
		auto dj = _DJ >> (H(s) + 1);
		auto dk = _DK >> (H(s) + 1);

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
		auto d1 = _D(_R(_Dual(s)));
		auto d2 = _D(_RR(_Dual(s)));
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
		auto di = _DI >> (H(s) + 1);
		auto dj = _DJ >> (H(s) + 1);
		auto dk = _DK >> (H(s) + 1);

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
		auto di = _DI >> (H(s) + 1);
		auto dj = _DJ >> (H(s) + 1);
		auto dk = _DK >> (H(s) + 1);

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

		auto d1 = _D(_R(s));
		auto d2 = _D(_RR(s));

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

		auto di = _DI >> (H(s) + 1);
		auto dj = _DJ >> (H(s) + 1);
		auto dk = _DK >> (H(s) + 1);

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

		auto di = _DI >> (H(s) + 1);
		auto dj = _DJ >> (H(s) + 1);
		auto dk = _DK >> (H(s) + 1);

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

		auto d1 = _D(_R(s));
		auto d2 = _D(_RR(s));

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

		auto d = _D(_Dual(s));
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
