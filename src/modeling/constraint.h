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
template<typename, int> class Geometry;
template<typename, typename > class Field;

template<typename TF>
class Constraint
{
public:
	typedef typename TF::mesh_type mesh_type;

	typedef typename mesh_type::index_type index_type;

	typedef typename mesh_type::coordinates_type coordinates_type;

	mesh_type const & mesh;

private:

	std::map<index_type, coordinates_type> def_domain_;

public:

	Constraint(mesh_type const & m)
			: mesh(m)
	{
	}

	~Constraint()
	{
	}

	std::map<index_type, coordinates_type> &GetDefDomain()
	{
		return def_domain_;
	}

	void SetDefDomain(std::function<bool(index_type, coordinates_type *)> const & selector)
	{
		mesh.SerialTraversal(TF::IForm, [&](index_type s)
		{
			coordinates_type coords;

			if(selector(s,&coords)) def_domain_.push_back(std::make_pair(s, coords));
		});

	}

	void SetDefDomain(std::function<bool(index_type)> const & selector)
	{
		mesh.SerialTraversal(TF::IForm, [&](index_type s)
		{
			if(selector(s )) def_domain_.push_back(std::make_pair(s, mesh.GetCoordinates(TF::IForm,s)));
		});

	}

	template<typename TDict>
	void SetDefDomain(TDict const & dict)
	{

		mesh.tags().template Select<TF::IForm>(dict, &def_domain_);

		if (def_domain_.empty())
		{
			WARNING << "Define domain is empty!";
		}
	}

	void Apply(TF * f, typename TF::value_type v) const
	{
		for (auto const & p : def_domain_)
		{
			(*f)[p.first] = v;
		}
	}

	typename std::enable_if<!std::is_same<typename TF::value_type, typename TF::field_value_type>::value, void>::type Apply(
	        TF * f, typename TF::field_value_type v) const
	{
		for (auto const & p : def_domain_)
		{
			(*f)[p.first] = mesh.template GetWeightOnElement<TF::IForm>(v, p.first);
		}
	}

	void Apply(TF * f, std::function<typename TF::value_type(coordinates_type const &, Real)> const & fun) const
	{
		for (auto const & p : def_domain_)
		{
			(*f)[p.first] = fun(p.second, mesh.GetTime());
		}
	}

	typename std::enable_if<!std::is_same<typename TF::value_type, typename TF::field_value_type>::value, void>::type Apply(
	        TF * f, std::function<typename TF::field_value_type(coordinates_type const &, Real)> const & fun) const
	{
		for (auto const & p : def_domain_)
		{
			(*f)[p.first] = mesh.template GetWeightOnElement<TF::IForm>(fun(p.second, mesh.GetTime()), p.first);
		}
	}

}
;

template<typename TField, typename TDict>
static std::function<void(TField *)> CreateConstraint(typename TField::mesh_type const & mesh, TDict const & dict)
{
	std::function<void(TField *)> res = [](TField *)
	{};

	std::shared_ptr<Constraint<TField>> self(new Constraint<TField>(mesh));

	self->SetDefDomain(dict["Select"]);
	CHECK(self->GetDefDomain().size());
	{
		auto value = dict["Value"];

		if (value.is_number())
		{
			auto foo = value.template as<typename TField::value_type>();

			res = [self,foo](TField * f )
			{	self->Apply(f,foo);};

		}
		else if (value.is_table())
		{
			auto foo = value.template as<typename TField::field_value_type>();

			res = [self,foo](TField * f )
			{	self->Apply(f,foo);};
		}
		else if (value.is_function())
		{
			std::function<typename TField::field_value_type(typename TField::coordinates_type, Real)> foo =
			        [value](typename TField::coordinates_type z, Real t)->typename TField::field_value_type
			        {
				        return value(z[0],z[1],z[2],t).template as<typename TField::field_value_type>();
			        };

			res = [self,foo](TField * f )
			{	self->Apply(f,foo);};
		}
	}

	return std::move(res);
}

}  // namespace simpla

#endif /* CONSTRAINT_H_ */
