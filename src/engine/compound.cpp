/*
 * compound.cpp
 *
 *  Created on: 2012-11-4
 *      Author: salmon
 */

#include "compound.h"
#include "include/simpla_defs.h"
#include "utilities/properties.h"
#include <boost/foreach.hpp>
namespace simpla
{
CompoundObject::CompoundObject()
{

}
CompoundObject::~CompoundObject()
{
}

//std::shared_ptr<CompoundObject> CompoundObject::Create(BaseContext * ctx,
//		ptree const & pt)
//{
//	std::shared_ptr<CompoundObject> res(new CompoundObject);
//
//	boost::optional<ptree const&> o_pt = pt.get_child_optional("<xmlattr>");
//
//	if (!!o_pt)
//	{
//		res->properties = *o_pt;
//
//	}
//	if (boost::optional<std::string> name = pt.get_optional<std::string>(
//			"<xmlattr>.Name"))
//
//	{
//		LOG << "Load Compound [" << *name << "]";
//	}
//	else
//	{
//		LOG << "Load an anonymous compound object";
//	}
//	for (auto &v : pt)
//	{
//		if (v.first == "<xmlcomment>" || v.first == "<xmlattr>")
//		{
//			continue;
//		}
//
//		boost::optional<std::string> o_name =
//				v.second.get_optional<std::string>("<xmlattr>.Name");
//
//		std::string o_type = v.first;
//
//		if (v.first == "Field" || v.first == "Pool")
//		{
//			o_type = o_type + "." + v.second.get("<xmlattr>.Type", "");
//		}
//
//		if (ctx->objFactory_.find(o_type) == ctx->objFactory_.end())
//		{
//			ERROR << "Unknown object type [" << o_type << "]";
//		}
//		else if (!o_name)
//		{
//			ERROR << "Can not register unnamed object !";
//		}
//
//		res->childs[*o_name] = ctx->objFactory_[o_type](
//				reinterpret_cast<ptree const&>(v.second));
//	}
//
//	return res;
//}

//std::shared_ptr<Object> CompoundObject::operator[](std::string const &name)
//{
//	return childs[name];
//}
//
//std::shared_ptr<Object> CompoundObject::operator[](
//		std::string const &name) const
//{
//	typename std::map<std::string, std::shared_ptr<Object> >::const_iterator it =
//			childs.find(name);
//	if (it == childs.end())
//	{
//		ERROR << "Can not find object " << name << " in the Compound!";
//	}
//	return it->second;
//}

boost::optional<std::shared_ptr<Object> > CompoundObject::Find(
		std::string const & name)
{
	if (name == "")
	{
		return boost::optional<std::shared_ptr<Object> >(false,
				std::shared_ptr<Object>());
	}

	std::map<std::string, std::shared_ptr<Object> >::iterator it = childs.find(
			name);

	return boost::optional<std::shared_ptr<Object> >(it != childs.end(),
			it->second);

}

boost::optional<std::shared_ptr<const Object> > CompoundObject::Find(
		std::string const & name) const
{
	if (name == "")
	{
		return boost::optional<std::shared_ptr<const Object> >(false,
				std::shared_ptr<const Object>());
	}

	std::map<std::string, std::shared_ptr<Object> >::const_iterator it =
			childs.find(name);

	return boost::optional<std::shared_ptr<const Object> >(it != childs.end(),
			it->second);

}
void CompoundObject::Add(std::string const & name, std::shared_ptr<Object> obj)
{
	;
}

void CompoundObject::Delete(std::string const & name)
{

	typename std::map<std::string, std::shared_ptr<Object> >::iterator it =
			childs.find(name);
	if (it != childs.end())
	{
		childs.erase(it);
	}

}
}  // namespace simpla
