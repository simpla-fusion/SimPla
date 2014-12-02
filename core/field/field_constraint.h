/*
 * field_constraint.h
 *
 *  Created on: 2014年12月2日
 *      Author: salmon
 */

#ifndef CORE_FIELD_FIELD_CONSTRAINT_H_
#define CORE_FIELD_FIELD_CONSTRAINT_H_
#include "../manifold/domain.h"
#include "../model/select.h"
namespace simpla
{
template<typename ...>struct Constraint;

template<typename TD, typename TV>
struct Constraint<TD, TV>
{

	typedef TD base_domain_type;

	typedef Constraint<TD, TV> this_type;

	typedef typename domain_traits<TD>::manifold_type manifold_type;

	static constexpr size_t iform = domain_traits<TD>::iform;

	typedef typename manifold_type::coordinates_type coordinates_type;

	typedef typename manifold_type::index_type index_type;

	typedef SubDomain<manifold_type, iform> domain_type;

	typedef TV value_type;
	typedef typename std::conditional<iform == EDGE || iform == FACE,
			nTuple<value_type, 3>, value_type>::type field_value_type;

	template<typename TDict>
	Constraint(std::shared_ptr<manifold_type> m, TDict const & dict) :
			domain_(Domain<manifold_type, iform>(m))
	{
		load(dict);
	}

	~Constraint()
	{
	}

	domain_type & domain()
	{
		return domain_;
	}
	domain_type const & domain() const
	{
		return domain_;
	}

	template<typename ...T>
	void operator()(_Field<T...> * f) const
	{
		if (!fun_)
		{
			VERBOSE << "Function is not defined! Do nothing!";
			return;
		}
		for (auto s : domain_)
		{
			auto x = domain_.coordinates(s);
			field_value_type v;
			fun_(x, &v);
			(*f)[s] = domain_.sample(s, v);
		}
	}
	template<typename TDict> void load(TDict const & dict);

private:

	domain_type domain_;

	std::function<void(coordinates_type, field_value_type *)> fun_;
};

template<typename TD, typename TV>
template<typename TDict>
void Constraint<TD, TV>::load(TDict const & dict)
{
	if (!dict["Select"] || !dict["Operation"])
	{
		return;
	}
	domain_ = select_by_config(
			dynamic_cast<Domain<manifold_type, iform> const&>(domain_),
			dict["Select"]);

	auto op = dict["Operation"];

	if (!dict["IsHardConstraint"])
	{
		fun_ = [=](coordinates_type x, field_value_type * v)
		{
			*v =op(x,*v);
		};
	}
	else
	{
		fun_ = [=](coordinates_type x, field_value_type * v)
		{
			*v =op(x);
		};
	}

}
template<size_t IFORM, typename TV, typename TM, typename TDict>
Constraint<Domain<TM, IFORM>, TV> make_constraint(std::shared_ptr<TM> const & m,
		TDict const & dict)
{
	return std::move(Constraint<Domain<TM, IFORM>, TV>(m, dict));
}
}
// namespace simpla

#endif /* CORE_FIELD_FIELD_CONSTRAINT_H_ */
