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

	typedef typename TF::value_type value_type;

	typedef typename mesh_type::index_type index_type;

	typedef typename mesh_type::coordinates_type coordinates_type;

	mesh_type const & mesh;

private:

	std::map<index_type, coordinates_type> def_domain_;

	std::function<void(value_type &, value_type)> op_;
public:

	Constraint(mesh_type const & m)
			: mesh(m)
	{
		SetHardSrc(false);
	}

	~Constraint()
	{
	}

	void SetHardSrc(bool flag = true)
	{
		if (flag)
		{
			op_ = [](value_type & a, value_type b)
			{	a =b;};
		}
		else
		{
			op_ = [](value_type & a, value_type b)
			{	a+=b;};
		}
	}

	std::map<index_type, coordinates_type> &GetDefDomain()
	{
		return def_domain_;
	}

	void Apply(TF * f, typename TF::value_type v) const
	{
		for (auto const & p : def_domain_)
		{
			op_((*f)[p.first], v);
		}
	}

	typename std::enable_if<!std::is_same<typename TF::value_type, typename TF::field_value_type>::value, void>::type Apply(
	        TF * f, typename TF::field_value_type v) const
	{
		for (auto const & p : def_domain_)
		{
			op_((*f)[p.first], mesh.template GetWeightOnElement<TF::IForm>(v, p.first));
		}
	}

	void Apply(TF * f, std::function<typename TF::value_type(coordinates_type const &, Real)> const & fun) const
	{
		for (auto const & p : def_domain_)
		{
			op_((*f)[p.first], fun(p.second, mesh.GetTime()));
		}
	}

	typename std::enable_if<!std::is_same<typename TF::value_type, typename TF::field_value_type>::value, void>::type Apply(
	        TF * f, std::function<typename TF::field_value_type(coordinates_type const &, Real)> const & fun) const
	{
		for (auto const & p : def_domain_)
		{
			op_((*f)[p.first], mesh.template GetWeightOnElement<TF::IForm>(fun(p.second, mesh.GetTime()), p.first));
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

	typedef typename TField::mesh_type::index_type index_type;
	typedef typename TField::mesh_type::coordinates_type coordinates_type;

	if (dict["Select"])
	{
		mesh.tags().template Select<TField::IForm>([&](index_type const &s ,coordinates_type const &x )
		{	self->GetDefDomain().emplace(s,x);},

		dict["Select"]);
	}
	else if (dict["Region"])
	{
		SelectFromMesh<TField::IForm>(mesh, [&](index_type const &s ,coordinates_type const &x )
		{	self->GetDefDomain().emplace(s,x);}, dict["Region"]);
	}
	else if (dict["Index"])
	{
		std::vector<nTuple<3, index_type>> idxs;
		dict["Index"].as(&idxs);
		SelectFromMesh<TField::IForm>(mesh, [&](index_type const &s ,coordinates_type const &x )
		{	self->GetDefDomain().emplace(s,x);}, idxs);
	}

	self->SetHardSrc(dict["HardSrc"].template as<bool>(false));

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
