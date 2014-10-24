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

namespace simpla
{
template<typename TCoordiantes = Real, typename TIndex = size_t>
struct DomainDummy: BlockRange<TIndex>
{
public:
	typedef TIndex index_type;
	typedef TCoordiantes coordinates_type;
	typedef BlockRange<TIndex> range_type;

	typedef DomainDummy<coordinates_type, index_type> this_type;

private:

	coordinates_type b_, e_, dx_;

public:

	template<typename ...Args>
	DomainDummy(Args && ... args) :
			range_type(std::forward<Args>(args)...)
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
		range_type::swap(that);
	}

	using range_type::hash;

	using range_type::max_hash;

	void coordinates_domain(coordinates_type b, coordinates_type e)
	{
		b_ = b;
		e_ = e;
		dx_ = (e_ - b_) / range_type::max_hash();
	}

	std::pair<coordinates_type, coordinates_type> coordinates_domain() const
	{
		return std::make_pair(b_, e_);
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

};

}
// namespace simpla

#endif /* DOMAIN_DUMMY_H_ */
