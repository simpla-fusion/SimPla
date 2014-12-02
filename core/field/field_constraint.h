/*
 * field_constraint.h
 *
 *  Created on: 2014年12月2日
 *      Author: salmon
 */

#ifndef CORE_FIELD_FIELD_CONSTRAINT_H_
#define CORE_FIELD_FIELD_CONSTRAINT_H_

namespace simpla
{
template<typename ...>struct FieldConstraint;

template<typename TD, typename TC>
struct FieldConstraint<_Field<TD, TC>>
{

	typedef _Field<TD, TC> field_type;
	typedef TD domain_type;
	typedef typename domain_type::coordinates_type coordinates_type;
	typedef typename domain_type::index_type index_type;
	typedef typename field_traits<field_type>::field_value_type field_value_type;

	SubDomain<TD> domain_;
	std::function<field_value_type(coordinates_type, field_value_type)> fun_;

	void operator()(_Field<TD, TC> *f) const
	{
		for (auto s : domain_)
		{
			auto x = domain_.coordinates(s);

			auto v = .template as<field_value_type>();

			(*f)[s] = domain_.sample(s, fun_(x, (*f)(x)));
		}
	}
};

}
// namespace simpla

#endif /* CORE_FIELD_FIELD_CONSTRAINT_H_ */
