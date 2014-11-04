/*
 * domain_dummy.h
 *
 *  Created on: 2014年10月29日
 *      Author: salmon
 */

#ifndef CORE_MANIFOLD_DOMAIN_DUMMY_H_
#define CORE_MANIFOLD_DOMAIN_DUMMY_H_

#include <stddef.h>
#include <algorithm>
#include "../utilities/primitives.h"
#include "../utilities/sp_type_traits.h"
#include "../parallel/block_range.h"

namespace simpla
{
template<typename ...>class _Field;
template<typename ...>class Expression;

template<typename TCoordiantes = Real, typename TIndex = size_t>
struct DomainDummy: BlockRange<TIndex>
{
public:
	typedef TIndex index_type;
	typedef TCoordiantes coordinates_type;

	typedef BlockRange<TIndex> range_type;

	typedef DomainDummy<coordinates_type, index_type> this_type;
	static constexpr size_t ndims = 1;
	static constexpr size_t iform = VERTEX;

	typedef this_type manifold_type;
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
	template<typename ...Args>
	void dimensions(Args && ... args)
	{
		this_type(std::forward<Args>(args)...).swap(*this);
	}
	// interpolator
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

	// topology
	template<typename ... Args>
	int dataset_shape(Args &&...args) const
	{
	}
	// diff_scheme
private:
	template<typename TOP, typename TL>
	inline auto calculate(TOP op, TL const& f, index_type s) const
	DECL_RET_TYPE( op(get(f,s) ) )

	template<typename TOP, typename TL, typename TR>
	inline auto calculate(TOP op, TL const& l, TR const &r, index_type s) const
	DECL_RET_TYPE( op(get( (l),s),get(r,s) ) )

public:

	template<typename TC, typename TD>
	inline auto get(_Field<TC, TD> const& f, index_type s) const
	DECL_RET_TYPE ((f[s]))

	template<typename TOP, typename TL>
	auto get(_Field<Expression<TOP, TL> > const & f, index_type s) const
	DECL_RET_TYPE((calculate(f.op_,f.lhs,s)))

	template<typename TOP, typename TL, typename TR>
	auto get(_Field<Expression<TOP, TL, TR> > const & f, index_type s) const
	DECL_RET_TYPE((calculate(f.op_,f.lhs,f.rhs,s)))

	template<typename T>
	auto get(T const & v, index_type s) const
	DECL_RET_TYPE((get_value(v,s)))

};

}
// namespace simpla

#endif /* CORE_MANIFOLD_DOMAIN_DUMMY_H_ */
