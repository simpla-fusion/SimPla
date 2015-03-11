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
template<typename ...>struct Constraint;

template<typename TM, typename TField>
struct Constraint<TM, TField>
{

	typedef TM mesh_type;

	typedef TField field_type;

	typedef Constraint<mesh_type> this_type;

	typedef typename mesh_type::coordinates_type coordinates_type;

	typedef typename mesh_type::id_type id_type;

	typedef typename field_type::value_type value_type;

	typedef typename field_type::field_value_type field_value_type;

	typedef std::function<
			field_value_type(Real, coordinates_type const &,
					field_value_type const &)> function_type;

	Constraint(mesh_type const &m)
			: m_mesh_(m)
	{
	}

	template<typename TFun>
	Constraint(mesh_type const &m, TFun const & fun = TFun())
			: m_mesh_(m), m_fun_(fun)
	{
	}

	Constraint(this_type const &other)
			: m_mesh_(other.m_mesh_), m_range_(other.m_range_), m_fun_(
					other.m_fun_)
	{
	}

	~Constraint()
	{
	}

	void swap(Constraint & other)
	{
		std::swap(m_mesh_, other.m_mesh_);
		std::swap(m_fun_, other.m_fun_);
		std::swap(m_range_, other.m_range_);

	}

	void operator()(field_type * f) const
	{
		if (m_range_.size() <= 0)
		{
			apply(m_mesh_.range(), f);
		}
		else
		{
			apply(m_range_, f);
		}
	}

	std::vector<id_type> & range()
	{
		return m_range_;
	}

	template<typename TRange>
	void range(TRange const & o_range)
	{
		for (auto s : o_range)
		{
			m_range_.push_back(s);
		}

	}
	template<typename TF>
	void function(TF const & f)
	{
		m_fun_ = f;
	}
	function_type const & function() const
	{
		return m_fun_;
	}

private:

	mesh_type m_mesh_;

	std::vector<id_type> m_range_;

	function_type m_fun_;

	template<typename TR>
	void apply(TR const & r, field_type * f) const
	{
		for (auto s : r)
		{
			auto x = m_mesh_.coordinates(s);

			Real t = m_mesh_.time();
//
			(*f)[s] = m_mesh_.sample(m_fun_(1, x, (*f)(x)), s);
		}
	}
};

template<typename TField, typename TM, typename TDict>
Constraint<TM, TField> make_constraint(TM const & mesh, TDict const & dict)
{
	typedef typename TM::coordinates_type coordinates_type;

	Constraint<TM, TField> res(mesh);

	typedef typename Constraint<TM, TField>::function_type function_type;

	typedef typename TField::field_value_type field_value_type;

	if (dict["Select"] && !dict["Operation"])
	{

//	res.range(select_by_config(mesh, dict["Select"]));

		auto op = dict["Operation"];

		if (!dict["IsHardConstraint"])
		{

			res.function(
					[=]( Real t,coordinates_type const & x, field_value_type const & v)->field_value_type
					{
						return op(t,x, v).template as<field_value_type>();
					}

					);
		}
		else
		{

			res.function(
					[=]( Real t,coordinates_type const &x, field_value_type const & v)->field_value_type
					{
						return op(t,x).template as<field_value_type>();
					});

		}
	}
	return std::move(res);
}

}
// namespace simpla

#endif /* CORE_FIELD_FIELD_CONSTRAINT_H_ */
