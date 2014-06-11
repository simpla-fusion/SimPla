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

template<typename TM, int IFORM, typename TV, typename TDict>
std::function<void()> CreateCommand(Field<TM, IFORM, TV>* f, TDict const & dict)
{
	std::function<void()> res = []()
	{};
	if (dict["Operation"] && dict["Select"])
	{
		auto def_domain = Filter(f->mesh.Select(IFORM), f->mesh, dict["Select"]);

		typedef TM mesh_type;

		typedef Field<TM, IFORM, TV> field_value_type;

		typedef typename mesh_type::iterator iterator;

		typedef typename mesh_type::coordinates_type coordinates_type;

		typedef std::function<field_value_type(Real, coordinates_type const &, field_value_type const &)> field_fun;

		auto op_ = dict["Operation"].template as<field_fun>();

		res = [=]()
		{
			for(auto s:def_domain)
			{
				auto x = f->mesh.GetCoordinates(s);

				(*f)[s] = f->mesh.Sample(Int2Type<IFORM>(), s, op_(f->mesh.GetTime(), x, (*f)(x)));
			}
		};

	}
	else
	{
		ERROR << "illegal configure! ";
	}
	return res;
}

}  // namespace simpla

#endif /* COMMAND_H_ */
