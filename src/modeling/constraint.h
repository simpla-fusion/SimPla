/*
 * constraint.h
 *
 *  Created on: 2014年2月13日
 *      Author: salmon
 */

#ifndef CONSTRAINT_H_
#define CONSTRAINT_H_
#include "../utilities/visitor.h"

namespace simpla
{

template<typename TF>
class Constraint: public VisitorBase
{
public:

	typedef TF field_type;

	static constexpr unsigned int IForm = field_type::IForm;

	typedef typename field_type::mesh_type mesh_type;

	typedef typename field_type::field_value_type field_value_type;

	typedef typename mesh_type::index_type index_type;

	typedef typename mesh_type::coordinates_type coordinates_type;

	mesh_type const & mesh;

private:

	std::list<index_type> def_domain_;
public:
	std::function<field_value_type(Real, coordinates_type, field_value_type const &)> op_;

	Constraint(mesh_type const & m)
			: mesh(m)
	{
	}

	~Constraint()
	{
	}

	void Insert(index_type s)
	{
		def_domain_.push_back(s);
	}
	bool empty() const
	{
		return def_domain_.empty();
	}
	void Visit(void * pf) const
	{
		// NOTE this is a danger opertaion , no type check

		field_type & f = *reinterpret_cast<field_type*>(pf);

		for (auto s : def_domain_)
		{
			auto x = mesh.GetCoordinates(s);

			f[s] = mesh.Sample(Int2Type<IForm>(), s, op_(mesh.GetTime(), x, f(x)));
		}
	}

}
;

template<typename TField, typename TDict>
std::shared_ptr<VisitorBase> CreateConstraint(Material<typename TField::mesh_type> const & material, TDict const & dict)
{

	typedef typename TField::mesh_type mesh_type;

	mesh_type const & mesh = material.mesh;

	std::shared_ptr<Constraint<TField>> res(new Constraint<TField>(mesh));

	typedef typename mesh_type::index_type index_type;

	typedef typename mesh_type::coordinates_type coordinates_type;

	typedef typename TField::field_value_type field_value_type;

	if (dict["Select"])
	{
		FilterRange<typename mesh_type::Range> range;

		auto obj = dict["Select"];

		auto type_str = obj["Type"].template as<std::string>("");

		CHECK(type_str);

		if (type_str == "Range")
		{
			range = Filter(mesh.GetRange(TField::IForm), mesh, obj["Value"]);
		}
		else
		{
			range = material.Select(mesh.GetRange(TField::IForm), obj);
		}

		for (auto s : range)
		{
			res->Insert(s);
		}

	}

	if (!res->empty() && dict["Operation"])
	{

		auto op = dict["Operation"];

		if (op.is_number() || op.is_table())
		{
			auto value = op.template as<field_value_type>();

			res->op_ = [value](Real,coordinates_type,field_value_type )->field_value_type
			{
				return value;
			};

		}
		else if (op.is_function())
		{
			res->op_ = [op](Real t,coordinates_type x,field_value_type v)->field_value_type
			{
				return op( t,x ,v).template as<field_value_type>();
			};

		}
	}
	else
	{
		ERROR << "illegal configuration!";
	}

	return std::dynamic_pointer_cast<VisitorBase>(res);
}

}  // namespace simpla

#endif /* CONSTRAINT_H_ */
