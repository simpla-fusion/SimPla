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
#include <vector>

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

	typedef IteratorRange<Value> IteratorRangeType;

	Pool(size_t value_size = sizeof(Value)) :
			data_(), value_size_in_bytes(value_size), max_size(0), tail(0)
	{
		// initialize node lists
	}

	virtual ~Pool()
	{
	}

	void Init(size_t maxs)
	{
		max_size = maxs;
		tail = 0;
		data_(
				TR1::shared_ptr<ByteType>(
						reinterpret_cast<ByteType*>(operator new(
								max_size * value_size_in_bytes))));

	}
	void Release()
	{
		TR1::shared_ptr<ByteType>().swap(data_);
		max_size = 0;
		tail = 0;
	}

	IteratorRangeType NewIteratorRange(size_t size)
	{

		size_t b = tail, e = (tail + size > max_size) ? max_size : tail + size;
		tail = e;
#ifdef _OMP

		size_t b1 = b+(e-b) * (omp_get_thread_num() / omp_get_num_threads());
		size_t e1 = b+(e-b) *((omp_get_thread_num() + 1 )/ omp_get_num_threads());
		b=b1;e=e1;
#endif
		return (IteratorRangeType(data_, b, e, value_size_in_bytes));
	}
	IteratorRangeType GetIteratorRange()
	{
		size_t b = 0, e = tail;
#ifdef _OMP
		size_t b1 = b+(e-b) * (omp_get_thread_num() / omp_get_num_threads());
		size_t e1 = b+(e-b) *((omp_get_thread_num() + 1 )/ omp_get_num_threads());
		b=b1;e=e1;
#endif
		return (IteratorRangeType(data_, b, e, value_size_in_bytes));
	}

// Metadata ------------------------------------------------------------

	virtual inline bool CheckType(std::type_info const & info) const
	{
		return (info == typeid(ThisType));
	}

	virtual bool IsEmpty() const
	{
		return (tail <= 0 || data_ == TR1::shared_ptr<ByteType>());
	}
private:
	TR1::shared_ptr<ByteType> data_;

	size_t max_size;

	size_t tail;

	const size_t value_size_in_bytes;

};
template<typename T>
class IteratorRange<Pool<T> >
{

public:
	typedef Pool<T> _Base;
	typedef T Value;
	typedef IteratorRange<_Base> ThisType;

	friend class _Base;

	IteratorRange(TR1::shared_ptr<Value> d, size_t begin, size_t end,
			size_t sb = sizeof(Value)) :
			base_(d), begin_(begin_), end_(end_), idx_(begin)
	{
	}
	~IteratorRange()
	{

	}

	bool IsEnd() const
	{
		return (idx_ >= end_);
	}

	ThisType &operator ++()
	{
		++idx_;
		return (*this);
	}
	ThisType operator ++(int)
	{
		ThisType res = *this;
		++idx_;
		return (res);
	}
	Value & operator *()
	{
		return (*reinterpret_cast<Value*>(&(*base_->data_)
				+ idx_ * base_->value_size_in_bytes));
	}

private:
	typename TR1::shared_ptr<_Base> base_;
	size_t idx_;
	size_t begin_;
	size_t end_;

};
} // namespace simpla
#endif /* PARTICLE_POOL_H_ */
