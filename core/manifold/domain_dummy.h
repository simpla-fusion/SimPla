/*
 * domain_dummy.h
 *
 *  Created on: 2014年10月3日
 *      Author: salmon
 */

#ifndef DOMAIN_DUMMY_H_
#define DOMAIN_DUMMY_H_

#include <stddef.h>
#include <algorithm>

#include "../parallel/block_range.h"
#include "../utilities/ntuple.h"

namespace simpla
{
template<size_t NDIMS = 1>
struct DomainDummy
{
public:

	static constexpr size_t ndims = NDIMS;
	static constexpr size_t iform = VERTEX;
	typedef DomainDummy<ndims> this_type;

	typedef nTuple<size_t, ndims> index_type;
	typedef nTuple<Real, ndims> coordinates_type;

private:

	coordinates_type b_, e_, dx_;
	index_type dims_;
	index_type strides_;
	size_t size_;
public:

	template<typename ...Args>
	DomainDummy(Args && ... args) :
			size_(0)
	{

	}
	~DomainDummy()
	{
	}

	void swap(this_type & that)
	{
		std::swap(b_, that.b_);
		std::swap(e_, that.e_);
		std::swap(dx_, that.dx_);
		std::swap(dims_, that.dims_);
		std::swap(strides_, that.strides_);
		std::swap(size_, that.size_);
	}

private:
	std::tuple<size_t, size_t> cal_strides(size_t dims)
	{
		return std::make_tuple(dims, 1);
	}
	std::tuple<size_t, nTuple<size_t, ndims> > cal_strides(
			nTuple<size_t, ndims> const & dims)
	{
		nTuple<size_t, ndims> strides;

		strides[ndims - 1] = 1;

		for (int i = ndims - 2; i >= 0; ++i)
		{
			strides[i] = dims[i] * strides[i + 1];
		}

		return std::make_tuple(dims[0] * strides[0], strides);
	}
	size_t dot_(size_t a, size_t b)
	{
		return a * b;
	}
	size_t dot_(nTuple<size_t, ndims> const & a,
			nTuple<size_t, ndims> const & b)
	{
		return dot(a, b);
	}
public:

	size_t max_hash() const
	{
		return size_;
	}
	size_t hash(index_type const &s) const
	{
		return dot_(s, strides_);
	}
	void extents(coordinates_type b, coordinates_type e)
	{
		b_ = b;
		e_ = e;
		dx_ = (e_ - b_) / dims_;
	}

	std::pair<coordinates_type, coordinates_type> extents() const
	{
		return std::make_pair(b_, e_);
	}
	void dimensions(index_type const &dims)
	{
		dims_ = dims;

		std::tie(size_, strides_) = cacl_strides(dims);

	}

	index_type const & dimensions() const
	{

		return dims_;
	}

	template<typename TD>
	auto gather(TD const & d,
			coordinates_type x) const->decltype(d[std::declval<index_type>()])
	{
		index_type r = static_cast<index_type>((x - b_) / dx_ + 0.5);

		return d[r];
	}

	template<typename TD, typename TV>
	void scatter(TD & d, coordinates_type x, TV const & v) const
	{
		index_type r = static_cast<index_type>((x - b_) / dx_ + 0.5);

		d[r] += v;
	}

	auto dataset() const
	DECL_RET_TYPE ((std::make_tuple(max_hash())))

};

}
// namespace simpla

#endif /* DOMAIN_DUMMY_H_ */
