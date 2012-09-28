/*
 * ndArray.h
 *
 *  Created on: 2011-12-25
 *      Author: salmon
 */

#ifndef NDARRAY_H_
#define NDARRAY_H_

/**
 *  class NdArray
 *   base class of Fields
 *   TODO:
 *   -) unstructured grid
 *   -) decomposing
 *   -) auto parallelism cuda/OpenMP
 *
 *
 * */
#include "include/simpla_defs.h"
#include "engine/object.h"
#include <vector>

namespace simpla
{
template<typename T>
class NdArray: public Object
{
public:
	typedef T ValueType;
	typedef NdArray<T> ThisType;
	typedef TR1::shared_ptr<ThisType> Holder;

	NdArray(std::vector<size_t> const& dims) :
			dimensions_(dims.begin(), dims.end()), strides_(dims.size())
	{
		int nd = dimensions_.size();
		strides_[nd - 1] = 1;
		for (int i = nd - 2; i >= 0; --i)
		{
			strides_[i] = strides_[i + 1] * dimensions_[i + 1];
		}

		size_t fullsize = strides_[0] * dimensions_[0];

		TR1::shared_ptr<ByteType>(new ByteType[fullsize]).swap(data_);
	}

	NdArray(ValueType* data, std::vector<size_t> const& dims,
			std::vector<size_t> const& s) :
			dimensions_(dims.begin(), dims.end()), strides_(s.begin(), s.end()), data_(
					data)
	{
	}

	virtual ~NdArray()
	{
	}

	void swap(ThisType & rhs)
	{
		dimensions_.swap(rhs.dimensions_);
		strides_.swap(rhs.strides_);
		data_.swap(rhs.data_);

	}
	static Holder create(std::vector<size_t> const& dims)
	{
		return (Holder(new NdArray(dims)));
	}

	bool operator !=(const NdArray & rhs)
	{
		return (data_ != rhs.data_);
	}

	inline bool empty() const
	{
		return (data_.get() == NULL);
	}
	inline ByteType* get_data()
	{
		return (data_.get());
	}
	inline int get_ele_size_in_bytes() const
	{
		return (*strides_.rbegin());
	}

	inline size_t get_size_in_bytes() const
	{
		return (strides_[0] * dimensions_[0]);
	}

	inline size_t get_num_of_element() const
	{
		return (strides_[0] * dimensions_[0] / (*strides_.rbegin()));
	}

	inline std::vector<size_t> const &
	get_dimensions() const
	{
		return (dimensions_);
	}
	inline std::vector<size_t> const &
	get_strides() const
	{
		return (strides_);
	}

	inline int get_num_of_dims() const
	{
		return (dimensions_.size());
	}

	void clear()
	{
		size_t size = this->get_size_in_bytes();
#pragma omp parallel for
		for (size_t s = 0; s < size; ++s)
		{
			data_[s] = 0;
		}
	}

	inline ValueType & get_value(size_t s)
	{
		return data_[s];
	}

	inline ValueType get_value(size_t s) const
	{
		return data_[s];
	}

	inline void set_value(size_t s, const ValueType & v)
	{
		data_[s] = v;
	}

	inline void add(size_t s, const ValueType & v)
	{

#pragma omp atomic
		data_[s] += v;
	}

private:

	TR1::shared_ptr<ValueType> data_;
	std::vector<size_t> dimensions_;
	std::vector<size_t> strides_;
}
;
} //namespace simpla
#endif /* NDndArray_H_ */
