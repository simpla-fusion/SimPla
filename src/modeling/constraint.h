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
template<typename, int, typename > class Field;

template<typename TF>
class Constraint
{
public:

	static constexpr unsigned int IForm = TF::IForm;

	typedef typename TF::mesh_type mesh_type;

	typedef typename TF::value_type value_type;

	typedef typename mesh_type::index_type index_type;

	typedef typename mesh_type::coordinates_type coordinates_type;

	mesh_type const & mesh;

private:

	std::list<index_type> def_domain_;

	std::function<void(value_type &, value_type)> op_;

	bool is_hard_src_;
public:

	Constraint(mesh_type const & m)
			: mesh(m), is_hard_src_(false)
	{
	}

	~Constraint()
	{
	}

	void SetHardSrc(bool flag = false)
	{
		is_hard_src_ = flag;
	}

	std::list<index_type> const &GetDefDomain() const
	{
		return def_domain_;
	}
	std::list<index_type> & GetDefDomain()
	{
		return def_domain_;
	}
	template<typename TV>
	void Apply(TF * f, TV v) const
	{
		for (auto const & s : def_domain_)
		{
			if (is_hard_src_)
			{
				f->get(s) = mesh.Sample(Int2Type<IForm>(), s, v);
			}
			else
			{
				f->get(s) += mesh.Sample(Int2Type<IForm>(), s, v);
			}
		}
	}
	template<typename TV>
	void Apply(TF * f, std::function<TV(coordinates_type, Real)> const & fun) const
	{
		for (auto const & s : def_domain_)
		{
			auto v = mesh.Sample(Int2Type<IForm>(), s, fun(mesh.GetCoordinates(s), mesh.GetTime()));

			if (is_hard_src_)
			{
				f->get(s) = v;
			}
			else
			{
				f->get(s) += v;
			}
		}
	}
}
;

template<typename TField, typename TDict>
static std::function<void(TField *)> CreateConstraint(Material<typename TField::mesh_type> const & material,
        TDict const & dict)
{
	std::function<void(TField *)> res = [](TField *)
	{};

	typedef typename TField::mesh_type mesh_type;

	mesh_type const & mesh = material.mesh;

	std::shared_ptr<Constraint<TField>> self(new Constraint<TField>(mesh));

	typedef typename mesh_type::index_type index_type;

	typedef typename mesh_type::coordinates_type coordinates_type;

	Range<IteratorFilter<mesh_type>> range(mesh, mesh.begin(TField::IForm), mesh.end(TField::IForm));

	if (dict["Select"])
	{
		range = material.template Select<TField::IForm>(dict["Select"]);
	}
	else if (dict["Range"])
	{
		range = SelectFromMesh(mesh, dict["Range"], mesh.being(TField::IForm), mesh.end(TField::IForm))
	}

	for (auto s : range)
	{
		self->GetDefDomain().push_back(s);
	}
//	else if (dict["Index"])
//	{
//		std::vector<nTuple<TField::mesh_type::NDIMS, size_t>> idxs;
//
//		dict["Index"].as(&idxs);
//
//		std::vector<typename TField::index_type> idx2;
//
//		for (auto const & v : idxs)
//		{
//			idx2.push_back(mesh.GetIndex(v));
//		}
//
//		SelectFromMesh<TField::IForm>(mesh, [&](index_type s )
//		{	self->GetDefDomain().push_back(s );}, idx2);
//
////		for (auto const &id : idxs)
////			self->GetDefDomain().push_back(mesh.GetIndex(id));
//	}
//	else
//	{
//	}

	self->SetHardSrc(dict["IsHard"].template as<bool>(false));

	if (dict["Value"])
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
