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

	typedef std::list<typename TM::iterator> define_domain_type;

	typedef Command<field_type> this_type;

	static constexpr unsigned int IForm = field_type::IForm;

	typedef typename field_type::mesh_type mesh_type;

	typedef typename field_type::field_value_type field_value_type;

	typedef typename mesh_type::iterator iterator;

	typedef typename mesh_type::coordinates_type coordinates_type;

	typedef std::function<field_value_type(Real, coordinates_type, field_value_type const &)> function_type;

	template<typename TDict, typename ...Others>
	Command(TDict dict, Others const & ...);
	~Command();

	template<typename ...Others>
	static std::function<void()> Create(field_type* f, Others const & ...);

	template<typename ... Others>
	static std::function<void(field_type*)> Create(Others const & ...);

	void Visit(field_type * f) const
	{
		LOGGER << "Apply field constraints";

		if (def_domain_.empty())
		{
			for (auto s : def_domain_)
			{
				auto x = f->mesh.GetCoordinates(s);

				(*f)[s] = f->mesh.Sample(Int2Type<IForm>(), s, op_(f->mesh.GetTime(), x, (*f)(x)));
			}
		}
		else
		{
			for (auto s : f->mesh.GetRange(IForm))
			{
				auto x = f->mesh.GetCoordinates(s);

				(*f)[s] = f->mesh.Sample(Int2Type<IForm>(), s, op_(f->mesh.GetTime(), x, (*f)(x)));
			}
		}
	}

private:
	define_domain_type def_domain_;

	function_type op_;

	void Visit_(void * pf) const
	{
		Visit(reinterpret_cast<field_type*>(pf));
	}
}
;

template<typename TM, int IFORM, typename TV>
template<typename TDict, typename ...Others>
Command<Field<TM, IFORM, TV>>::Command(TDict dict, Others const & ...others)
{
	Select(&def_domain_, dict["Select"], std::forward<Others const &>(others)...);

	if (dict["Operation"])
	{
		op_ = dict["Operation"].template as<function_type>();
	}
	else
	{
		ERROR << "illegal configure! ";
	}

}

template<typename TM, int IFORM, typename TV>
Command<Field<TM, IFORM, TV>>::~Command()
{
}

template<typename TM, int IFORM, typename TV>
template<typename ... Others>
std::function<void()> Command<Field<TM, IFORM, TV> >::Create(field_type* f, Others const & ...others)
{

	return std::bind(&this_type::Visit,
	        std::shared_ptr<this_type>(new this_type(std::forward<Others const &>(others)...)), f);
}
template<typename TM, int IFORM, typename TV>
template<typename ... Others>
std::function<void(Field<TM, IFORM, TV>*)> Command<Field<TM, IFORM, TV> >::Create(Others const & ...others)
{

	return std::bind(&this_type::Visit,
	        std::shared_ptr<this_type>(new this_type(std::forward<Others const &>(others)...)), std::placeholders::_1);
}

}  // namespace simpla

#endif /* COMMAND_H_ */
