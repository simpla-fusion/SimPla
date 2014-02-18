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

template<typename TM, int IFORM>
class Constraint
{
public:
	typedef TM mesh_type;

	typedef typename mesh_type::index_type index_type;

	typedef typename mesh_type::coordinates_type coordinates_type;

	mesh_type const & mesh;

private:

	std::map<index_type, coordinates_type> def_domain_;

public:

	Constraint(mesh_type const & m) :
			mesh(m)
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
		mesh.SerialTraversal(IFORM, [&](index_type s)
		{
			coordinates_type coords;

			if(selector(s,&coords)) def_domain_.push_back(std::make_pair(s, coords));
		});

	}

	void SetDefDomain(std::function<bool(index_type)> const & selector)
	{
		mesh.SerialTraversal(IFORM, [&](index_type s)
		{
			if(selector(s )) def_domain_.push_back(std::make_pair(s, mesh.GetCoordinates(IFORM,s)));
		});

	}

	template<typename TDict>
	void SetDefDomain(TDict const & dict)
	{

		mesh.tags().template Select<IFORM>(dict["Select"], &def_domain_);

		if (def_domain_.empty())
		{
			WARNING << "Define domain is empty!";
		}
	}

	template<typename TField, typename TDict>
	static std::function<void(TField *)> Create(mesh_type const & mesh, TDict const & dict)
	{
		std::function<void(TField *)> res;

		std::shared_ptr<Constraint<mesh_type, IFORM>> self(new Constraint<mesh_type, IFORM>(mesh));

		self->SetDefDomain(dict["Select"]);

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
				std::function<typename TField::field_value_type(coordinates_type, Real)> foo =
						[value](coordinates_type z, Real t)->typename TField::field_value_type
						{
							return value(z[0],z[1],z[2],t).template as<typename TField::field_value_type>();
						};

				res = [self,foo](TField * f )
				{	self->Apply(f,foo);};
			}
		}

		return std::move(res);
	}

	template<typename TV> void Apply(Field<Geometry<mesh_type, IFORM>, TV> * f,
			typename Field<Geometry<mesh_type, IFORM>, TV>::value_type v)
	{
		for (auto const & p : def_domain_)
		{
			(*f)[p.first] = v;
		}
	}

	template<typename TV> typename std::enable_if<
			!std::is_same<typename Field<Geometry<mesh_type, IFORM>, TV>::value_type,
					typename Field<Geometry<mesh_type, IFORM>, TV>::field_value_type>::value, void>::type Apply(
			Field<Geometry<mesh_type, IFORM>, TV> * f,
			typename Field<Geometry<mesh_type, IFORM>, TV>::field_value_type v)
	{
		for (auto const & p : def_domain_)
		{
			(*f)[p.first] = mesh.template GetWeightOnElement<IFORM>(v, p.first);
		}
	}

	template<typename TV>
	void Apply(Field<Geometry<mesh_type, IFORM>, TV> * f,
			std::function<typename Field<Geometry<mesh_type, IFORM>, TV>::value_type(coordinates_type const &, Real)> const & fun)
	{
		for (auto const & p : def_domain_)
		{
			(*f)[p.first] = fun(p.second, mesh.GetTime());
		}
	}

	template<typename TV> typename std::enable_if<
			!std::is_same<typename Field<Geometry<mesh_type, IFORM>, TV>::value_type,
					typename Field<Geometry<mesh_type, IFORM>, TV>::field_value_type>::value, void>::type Apply(
			Field<Geometry<mesh_type, IFORM>, TV> * f,
			std::function<
					typename Field<Geometry<mesh_type, IFORM>, TV>::field_value_type(coordinates_type const &, Real)> const & fun)
	{
		for (auto const & p : def_domain_)
		{
			(*f)[p.first] = mesh.template GetWeightOnElement<IFORM>(fun(p.second, mesh.GetTime()), p.first);
		}
	}

}
;

}  // namespace simpla

#endif /* CONSTRAINT_H_ */
