/*
 * particle_pool.h
 *
 *  Created on: 2012-3-15
 *      Author: salmon
 */

#ifndef PARTICLE_POOL_H_
#define PARTICLE_POOL_H_
#include "include/simpla_defs.h"
#include "engine/object.h"
#include <openmp.h>
#include <iterator>

namespace simpla
{

template<typename T> class IteratorRange;

template<typename T>
struct Pool: public Object
{

public:
	//TODO Define a parallel iterator

	typedef T Value;

	typedef Pool<Value> ThisType;

	Pool(size_t n, size_t s = sizeof(Value)) :
			max_num_(n),

			value_size_in_bytes_(s),

			data_(
					std::shared_ptr<ByteType>(
							reinterpret_cast<ByteType*>(operator new(
									max_num_ * value_size_in_bytes_)))),

			begin_(data_, value_size_in_bytes_),

			end_(data_, value_size_in_bytes_,
					data_.get() + max_num_ * value_size_in_bytes_)

	{
	}

	virtual ~Pool()
	{
	}

	class iterator: public std::iterator<std::input_iterator_tag, T,
			std::ptrdiff_t, T*, T&>
	{
		size_t value_size_in_bytes_;
		std::shared_ptr<ByteType> data_;
		T* p_;

	public:

		iterator(std::shared_ptr<ByteType> d, size_t s = sizeof(T),
				ByteType * p = nullptr) :
				value_size_in_bytes_(s), data_(d), p_(
						p == nullptr ? d.get() : p)
		{
		}

		~iterator()
		{
		}

		T & operator*()
		{
			return (*reinterpret_cast<T*>(p_));
		}
		T const& operator*() const
		{
			return (*reinterpret_cast<T const*>(p_));
		}
		iterator & operator++()
		{
			p_ += value_size_in_bytes_;
			return *this;
		}

		bool operator==(iterator const & r)
		{
			return (p_ == r.p_);
		}
	};

	class const_iterator: public std::iterator<std::output_iterator_tag,
			const T, std::ptrdiff_t, T const*, T const&>
	{
		std::shared_ptr<const T> data_;
		T const* p_;
		size_t value_size_in_bytes_;
	public:

		const_iterator(std::shared_ptr<const ByteType> d, size_t s = sizeof(T),
				ByteType const * p = nullptr) :
				value_size_in_bytes_(s), data_(d), p_(
						p == nullptr ? d.get() : p)
		{
		}

		~const_iterator()
		{
		}

		T const& operator*() const
		{
			return (*reinterpret_cast<T const*>(p_));
		}
		const_iterator & operator++()
		{
			p_ += value_size_in_bytes_;
			return *this;
		}
	};

	inline iterator begin()
	{
		return begin_;
	}
	inline iterator end()
	{
		return end_;
	}

	inline const_iterator begin() const
	{
		return begin_;
	}
	inline iterator end() const
	{
		return end_;
	}

//// Metadata ------------------------------------------------------------
//
//	virtual inline bool CheckType(std::type_info const & info) const
//	{
//		return (info == typeid(ThisType));
//	}
//
//	virtual bool IsEmpty() const
//	{
//		return (tail <= 0 || data_ == std::shared_ptr<ByteType>());
//	}
//
//
//	Pool NewIteratorRange(size_t size)
//	{
//
//		size_t b = tail, e = (tail + size > max_size) ? max_size : tail + size;
//		tail = e;
//#ifdef _OMP
//
//		size_t b1 = b+(e-b) * (omp_get_thread_num() / omp_get_num_threads());
//		size_t e1 = b+(e-b) *((omp_get_thread_num() + 1 )/ omp_get_num_threads());
//		b=b1;e=e1;
//#endif
//		return (IteratorRangeType(data_, b, e, value_size_in_bytes));
//	}
//	Pool GetIteratorRange()
//	{
//		size_t b = 0, e = tail;
//#ifdef _OMP
//		size_t b1 = b+(e-b) * (omp_get_thread_num() / omp_get_num_threads());
//		size_t e1 = b+(e-b) *((omp_get_thread_num() + 1 )/ omp_get_num_threads());
//		b=b1;e=e1;
//#endif
//		return (IteratorRangeType(data_, b, e, value_size_in_bytes));
//	}
private:
	const size_t value_size_in_bytes_;

	size_t max_num_;

	std::shared_ptr<ByteType> data_;

	iterator begin_;
	iterator end_;

};

} // namespace simpla
#endif /* PARTICLE_POOL_H_ */
