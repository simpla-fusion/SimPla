/*
 * command.h
 *
 *  Created on: 2014年4月25日
 *      Author: salmon
 */

#ifndef COMMAND_H_
#define COMMAND_H_

#include <memory>
#include <functional>
#include <list>
#include <string>

#include "select.h"
#include "../utilities/log.h"
#include "../utilities/type_utilites.h"
#include "../utilities/visitor.h"
#include "../fetl/fetl.h"
namespace simpla
{
template<typename TF> class Command;

template<typename TM, int IFORM, typename TV>
class Command<Field<TM, IFORM, TV> > : public VisitorBase
{
public:

	typedef Field<TM, IFORM, TV> field_type;

	typedef std::list<typename TM::index_type> define_domain_type;

	typedef Command<field_type> this_type;

	static constexpr unsigned int IForm = field_type::IForm;

	typedef typename field_type::mesh_type mesh_type;

	typedef typename field_type::field_value_type field_value_type;

	typedef typename mesh_type::index_type index_type;

	typedef typename mesh_type::coordinates_type coordinates_type;

	mesh_type const & mesh;

private:

	define_domain_type def_domain_;
public:
	std::function<
			field_value_type(Real, coordinates_type, field_value_type const &)> op_;

	template<typename TDict, typename TModel, typename ...Others>
	Command(TDict const & dict, TModel const &model, Others const & ...);
	~Command();

	template<typename ...Others>
	static std::function<void()> Create(field_type* f, Others const & ...);

	template<typename ... Others>
	static std::function<void(field_type*)> Create(Others const & ...);

	void Visit(field_type * f) const
	{
		// NOTE this is a danger opertaion , no type check

		for (auto s : def_domain_)
		{
			auto x = mesh.GetCoordinates(s);

			(*f)[s] = mesh.Sample(Int2Type<IForm>(), s,
					op_(mesh.GetTime(), x, (*f)(x)));
		}
	}

private:
	void Visit_(void * pf) const
	{
		Visit(reinterpret_cast<field_type*>(pf));
	}
}
;

template<typename TM, int IFORM, typename TV>
template<typename TDict, typename TModel, typename ...Others>
Command<Field<TM, IFORM, TV>>::Command(TDict const & dict, TModel const &model,
		Others const & ...) :
		mesh(model.mesh)
{

	if (dict["Select"])
	{
		FilterRange<typename mesh_type::Range> range;

		auto obj = dict["Select"];

		auto type_str = obj["Type"].template as<std::string>();

		if (type_str == "Range")
		{
			range = Filter(mesh.GetRange(IForm), mesh, obj["Value"]);
		}
		else
		{
			range = model.Select(mesh.GetRange(IForm), obj);
		}

		for (auto s : range)
		{
			def_domain_.push_back(s);
		}

	}

	if (!def_domain_.empty() && dict["Operation"])
	{

		auto op = dict["Operation"];

		if (op.is_number() || op.is_table())
		{
			auto value = op.template as<field_value_type>();

			op_ =
					[value](Real,coordinates_type,field_value_type )->field_value_type
					{
						return value;
					};

		}
		else if (op.is_function())
		{
			op_ =
					[op](Real t,coordinates_type x,field_value_type v)->field_value_type
					{	return op( t,x ,v).template as<field_value_type>();
					};

		}
	}
	else
	{
		ERROR << "illegal configuration!";
	}

}

template<typename TM, int IFORM, typename TV>
Command<Field<TM, IFORM, TV>>::~Command()
{
}

template<typename TM, int IFORM, typename TV>
template<typename ... Others>
std::function<void()> Command<Field<TM, IFORM, TV> >::Create(field_type* f,
		Others const & ...others)
{

	return std::bind(&this_type::Visit,
			std::shared_ptr<this_type>(
					new this_type(std::forward<Others const &>(others)...)), f);
}
template<typename TM, int IFORM, typename TV>
template<typename ... Others>
std::function<void(Field<TM, IFORM, TV>*)> Command<Field<TM, IFORM, TV> >::Create(
		Others const & ...others)
{

	return std::bind(&this_type::Visit,
			std::shared_ptr<this_type>(
					new this_type(std::forward<Others const &>(others)...)),
			std::placeholders::_1);
}

}  // namespace simpla

#endif /* COMMAND_H_ */
