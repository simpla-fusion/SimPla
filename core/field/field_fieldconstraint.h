/*
 * @file field_constraint.h
 *
 *  Created on: 2014年12月2日
 *      Author: salmon
 */

#ifndef CORE_FIELD_FIELD_FIELDCONSTRAINT_H_
#define CORE_FIELD_FIELD_FIELDCONSTRAINT_H_
#include "../model/select.h"
#include "../utilities/log.h"
#include <set>
namespace simpla
{

template<typename TField>
struct FieldConstraint
{
	typedef TField field_type;

	typedef FieldConstraint<field_type> this_type;

	typedef typename field_type::mesh_type mesh_type;

	typedef typename mesh_type::coordinates_type coordinates_type;

	typedef typename mesh_type::id_type id_type;

	typedef typename field_type::field_value_type field_value_type;

	typedef std::function<
			field_value_type(coordinates_type const &, Real,
					field_value_type const &)> function_type;

	typedef std::set<id_type> range_type;

	FieldConstraint(mesh_type const &m, bool hard = false)
			: m_mesh_(m), m_is_hard_(hard)
	{
	}
	FieldConstraint(mesh_type const &m, function_type const & fun, bool hard =
			false)
			: m_mesh_(m), m_fun_(fun), m_is_hard_(hard)
	{
	}

	FieldConstraint(mesh_type const &m, function_type const & fun,
			range_type const & r, bool hard = false)
			: m_mesh_(m), m_fun_(fun), m_range_(r), m_is_hard_(hard)
	{
	}

	FieldConstraint(this_type const &other)
			: m_mesh_(other.m_mesh_), m_range_(other.m_range_), m_fun_(
					other.m_fun_)
	{
	}

	~FieldConstraint()
	{
	}

	void swap(FieldConstraint & other)
	{
		std::swap(m_mesh_, other.m_mesh_);
		std::swap(m_fun_, other.m_fun_);
		std::swap(m_range_, other.m_range_);
	}

	bool m_is_hard_ = true;

	bool is_hard() const
	{
		return m_is_hard_;
	}

	void is_hard(bool p_is_hard)
	{
		m_is_hard_ = p_is_hard;
	}

	range_type & range()
	{
		return m_range_;
	}

	function_type const & function() const
	{
		return m_fun_;
	}
	template<typename TFun>
	void function(TFun const & foo)
	{
		m_fun_ = foo;
	}
	template<typename TF>
	void operator()(TF * f) const
	{

		if (!m_fun_)
			return;

		if (m_range_.size() <= 0)
		{
			apply(m_mesh_.range(), f);
		}
		else
		{
			apply(m_range_, f);
		}
	}
private:

	mesh_type m_mesh_;

	std::set<id_type> m_range_;

	function_type m_fun_;

	template<typename TDomain>
	void apply(TDomain const & r, field_type * f) const;

};
template<typename TField>
template<typename TDomain>
void FieldConstraint<TField>::apply(TDomain const & r, field_type * f) const
{
	if (!f->is_valid())
	{
		f->clear();
	}

	field_value_type v;

	for (auto s : r)
	{
		auto x = m_mesh_.coordinates(s);
		Real t = m_mesh_.time();
		if (!is_hard())
		{
			v = (*f)(x);
		}
		(*f)[s] = m_mesh_.sample(m_fun_(x, t, v), s);
	}

}

template<typename TField, typename TDict>
FieldConstraint<TField> make_constraint(typename TField::mesh_type const & mesh,
		TDict const & dict)
{

	FieldConstraint<TField> res(mesh);

	typedef TField field_type;

	typedef typename FieldConstraint<field_type>::function_type function_type;

	typedef typename field_type::field_value_type field_value_type;

	typedef typename field_type::mesh_type mesh_type;

	typedef typename mesh_type::coordinates_type coordinates_type;

	typedef typename mesh_type::id_type id_type;

	res.is_hard(dict["IsHard"]);

	if (dict["Domain"])
	{
		select_ids_by_config(mesh, dict["Domain"], mesh.range(), &res.range());
	}

	if (dict["Value"])
	{

		auto op = dict["Value"];

		if (op.is_function())
		{
			res.function(

			[=]( coordinates_type const & x,Real t,
					field_value_type const & f)->field_value_type
			{
				return op(x,t,f).template as<field_value_type>();
			}

			);
		}
		else
		{
			auto v = op.template as<field_value_type>();

			res.function(

			[=]( coordinates_type const & ,Real ,
					field_value_type const & )->field_value_type
			{
				return v;
			}

			);

		}

	}
	return std::move(res);
}
template<typename TField, typename TDict>
void apply_constraint(TDict const & dict, TField * f)
{
	make_constraint<TField>(f->mesh(), dict)(f);
}
}
// namespace simpla

#endif /* CORE_FIELD_FIELD_FIELDCONSTRAINT_H_ */
