/*
 * constraint.h
 *
 *  Created on: 2014年2月13日
 *      Author: salmon
 */

#ifndef CONSTRAINT_H_
#define CONSTRAINT_H_

namespace simpla
{

class ConstraintBase
{
public:
	ConstraintBase()
	{
	}
	virtual ~ConstraintBase()
	{

	}
	virtual void Apply(void* p) const=0;
};

template<typename TF>
class Constraint: public ConstraintBase
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

	std::list<index_type> const &GetDefDomain() const
	{
		return def_domain_;
	}
	std::list<index_type> & GetDefDomain()
	{
		return def_domain_;
	}

	void Apply(void * pf) const
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
std::function<void(TField *)> CreateConstraint(Material<typename TField::mesh_type> const & material,
        TDict const & dict)
{

	typedef typename TField::mesh_type mesh_type;

	mesh_type const & mesh = material.mesh;

	std::shared_ptr<Constraint<TField>> self(new Constraint<TField>(mesh));

	typedef typename mesh_type::index_type index_type;

	typedef typename mesh_type::coordinates_type coordinates_type;

	typedef typename TField::field_value_type field_value_type;

	FilterRange<typename mesh_type::Range> range;

	if (dict["Select"])
	{
		range = material.Select(mesh.GetRange(TField::IForm), dict["Select"]);
	}
	else if (dict["Range"])
	{
		range = Filter(mesh.GetRange(TField::IForm), mesh, dict["Range"]);
	}

	for (auto s : range)
	{
		self->GetDefDomain().push_back(s);
	}

	if (!self->GetDefDomain().empty())
	{

		if (dict["Value"])
		{
			auto obj = dict["Value"];

			if (obj.is_number() || obj.is_table())
			{
				auto value = obj.template as<field_value_type>();

				self->op_ = [value](Real,coordinates_type,field_value_type )->field_value_type
				{
					return value;
				};

			}
			else if (obj.is_function())
			{
				self->op_ = [obj](Real t,coordinates_type x,field_value_type v)->field_value_type
				{
					return obj( t,x ,v).template as<field_value_type>();
				};

			}
		}
	}
	else
	{
		WARNING << "Define domain is empty!";
	}
	std::function<void(TField *)> res = std::bind(&Constraint<TField>::Apply, self, std::placeholders::_1);

	return std::move(res);
}

}  // namespace simpla

#endif /* CONSTRAINT_H_ */
