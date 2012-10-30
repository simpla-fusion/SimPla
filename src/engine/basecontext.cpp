/*
 * domain.cpp
 *
 *  Created on: 2012-10-13
 *      Author: salmon
 */
#include <iostream>
#include "include/simpla_defs.h"
#include "context.h"
#include "object.h"
#include "fetl/grid.h"
#include "io/read_hdf5.h"
namespace simpla
{
BaseContext::BaseContext() :
		dt(0.0), counter_(0), timer_(0)
{

}
BaseContext::BaseContext(ptree const&pt) :
		dt(pt.get("Grid.Time.<xmlattr>.dt", 1.0d)),

		PHYS_CONSTANTS(pt.get_child("PhysConstants")),

		counter_(0), timer_(0)
{
}
BaseContext::~BaseContext()
{
}
//TR1::shared_ptr<BaseContext> BaseContext::Create(ptree const & pt)
//{
//	TR1::shared_ptr<BaseContext> res;
//
//	std::string topology = pt.get("Topology.<xmlattr>.Type", "CoRectMesh");
//
//	if (topology == "CoRectMesh")
//	{
//		res = TR1::shared_ptr<BaseContext>(new Context<UniformRectGrid>(pt))
//	}
//	else
//	{
//		ERROR << "Unregistered Context type!" << topology;
//	}
//
//	return res;
//
//}

boost::optional<TR1::shared_ptr<Object> > BaseContext::FindObject(
		std::string const & name, std::type_info const &tinfo)
{
	TR1::shared_ptr<Object> res;

	if (name == "")
	{
		return boost::optional<TR1::shared_ptr<Object> >();
	}
	std::map<std::string, Object::Holder>::iterator it = objects.find(name);

	return boost::optional<TR1::shared_ptr<Object> >(
			it != objects.end()
					&& (tinfo == typeid(void) || it->second->CheckType(tinfo)),
			it->second);

}

boost::optional<TR1::shared_ptr<const Object> > BaseContext::FindObject(
		std::string const & name, std::type_info const &tinfo) const
{

	if (name == "")
	{
		return boost::optional<TR1::shared_ptr<const Object> >();
	}
	std::map<std::string, Object::Holder>::const_iterator it = objects.find(
			name);

	return boost::optional<TR1::shared_ptr<const Object> >(
			it != objects.end()
					&& (tinfo == typeid(void) || it->second->CheckType(tinfo)),
			it->second);

}
void BaseContext::DeleteObject(std::string const & name)
{
	typename std::map<std::string, TR1::shared_ptr<Object> >::iterator it =
			objects.find(name);
	if (it != objects.end())
	{
		objects.erase(it);
	}

}

inline void eval_(TR1::function<void(void)> & f)
{
	f();
}
void BaseContext::Eval(size_t maxstep)
{

	for (size_t i = 0; i < maxstep; ++i)
	{
		LOG << "COUNTER:" << counter_ << " Time:" << timer_;
		std::for_each(modules.begin(), modules.end(), eval_);

		++counter_;
		timer_ += dt;
	}
}

void BaseContext::Load(ptree const & pt)
{

	BOOST_FOREACH(const typename ptree::value_type &v, pt.get_child("Grid"))
	{
		if (v.first != "Attribute")
		{
			continue;
		}
		std::string type = v.second.get<std::string>("<xmlattr>.Type");

		std::string id = v.second.get<std::string>("<xmlattr>.Name");

		if (objFactory_.find(type) != objFactory_.end())
		{
			objects[id] = objFactory_[type]();

			boost::optional<std::string> url =
					v.second.get_optional<std::string>("url");
			if (!!url)
			{
				io::ReadData(*url, objects[id]);
			}
			LOG << "Load data " << id << "<" << type << ">";
		}
		else
		{
			WARNING << "Object type " << type << " is not registered!";
		}

	}

	BOOST_FOREACH(const typename ptree::value_type &v, pt.get_child("Process"))
	{
		if (v.first != "Module")
		{
			continue;
		}
		std::string type = v.second.get<std::string>("<xmlattr>.Type");
		if (moduleFactory_.find(type) != moduleFactory_.end())
		{
			modules.push_back(moduleFactory_[type](v.second));
			LOG << "Add module " << type << " successed!";
		}
		else
		{
			WARNING << "Module type " << type << " is not registered!";
		}

	}
	modules.push_back(
			moduleFactory_[pt.get("Process.Output.<xmlattr>.Type", "XDMF")](
					pt.get_child("Process.Output")));
}

}  // namespace simpla

