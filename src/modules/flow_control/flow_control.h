/*
 * flow_control.h
 *
 *  Created on: 2012-10-31
 *      Author: salmon
 */

#ifndef FLOW_CONTROL_H_
#define FLOW_CONTROL_H_
#include "include/simpla_defs.h"
#include "engine/context.h"
#include "engine/modules.h"
#include "utilities/properties.h"
#include "io/read_hdf5.h"

namespace simpla
{

namespace flow_control
{

class Clock: public Module
{
public:
	typedef Clock ThisType;

	BaseContext & ctx;

	Clock(BaseContext & d, const ptree & pt) :
			ctx(d)
	{
	}

	virtual ~Clock()
	{
	}
	static TR1::function<void(void)> Create(BaseContext* d, const ptree & pt)
	{
		return TR1::bind(&ThisType::Eval,
				TR1::shared_ptr<ThisType>(new ThisType(*d, pt)));
	}

	virtual void Eval()
	{
		ctx.PushClock();

		LOG << "Counter: " << ctx.Counter() << "  Time: " << ctx.Timer();
	}

}
;

class LoadField
{
public:
	typedef LoadField ThisType;

	LoadField(BaseContext & d, const ptree & pt) :
			ctx(d), pt_(pt),

			name(pt.get<std::string>("<xmlattr>.Name")),

			type(pt.get<std::string>("<xmlattr>.Type"))
	{

		if (ctx.objFactory_.find(type) == ctx.objFactory_.end())
		{
			ERROR << "Object type " << type << " is not registered!";
		}
		ctx.objects[name] = ctx.objFactory_[type]();

	}

	virtual ~LoadField()
	{
	}
	static TR1::function<void(void)> Create(BaseContext* d, const ptree & pt)
	{
		return TR1::bind(&ThisType::Eval,
				TR1::shared_ptr<ThisType>(new ThisType(*d, pt)));
	}

	virtual void Eval()
	{

		TR1::shared_ptr<ArrayObject> obj =
				TR1::dynamic_pointer_cast<ArrayObject>(ctx.objects[name]);

		boost::optional<std::string> format = pt_.get_optional<std::string>(
				"Data.<xmlattr>.Format");

		if (!format)
		{
			obj->Clear();
		}
		else if (*format == "HDF")
		{
			std::string url = pt_.get<std::string>("Data");
			io::ReadData(url, obj);
		}
		else if (*format == "XML")
		{
			if (obj->CheckValueType(typeid(Integral)))
			{
				obj->FullFill(pt_.get<Integral>("Data", 0));
			}
			else if (obj->CheckValueType(typeid(Real)))
			{
				obj->FullFill(pt_.get("Data", 0.0d));
			}
			else if (obj->CheckValueType(typeid(Complex)))
			{
				Complex dv(0, 0);
				Complex a = pt_.get("Data", dv,
						pt_trans<Complex, std::string>());
				obj->FullFill(a);
			}
			else if (obj->CheckValueType(typeid(nTuple<THREE, Real> )))
			{
				nTuple<THREE, Real> dv =
				{ 0, 0, 0 };
				nTuple<THREE, Real> a = pt_.get("Data", dv,
						pt_trans<nTuple<THREE, Real>, std::string>());
				obj->FullFill(a);
			}
			else if (obj->CheckValueType(typeid(nTuple<THREE, Complex> )))
			{
				nTuple<THREE, Complex> dv =
				{ 0, 0, 0 };
				nTuple<THREE, Complex> a = pt_.get("Data", dv,
						pt_trans<nTuple<THREE, Complex>, std::string>());
				obj->FullFill(a);
			}

		}

		LOG << "Load data " << name << "<" << type << ">";

	}

private:

	BaseContext & ctx;
	ptree pt_;
	std::string name, type;

};

inline void eval_(TR1::function<void(void)> & f)
{
	f();
}
class Loop: public Module
{
public:
	typedef Loop ThisType;

	size_t maxstep;

	BaseContext & ctx;

	std::list<TR1::function<void(void)> > modules;

	Loop(BaseContext & d, const ptree & pt) :
			ctx(d), maxstep(pt.get("<xmlattr>.Steps", 1))
	{
		BOOST_FOREACH(const typename ptree::value_type &v, pt)
		{
			CHECK(v.first);
			if (v.first == "<xmlcomment>" || v.first == "<xmlattr>")
			{
				continue;
			}
			if (ctx.moduleFactory_.find(v.first) != ctx.moduleFactory_.end())
			{
				modules.push_back(ctx.moduleFactory_[v.first](v.second));
				LOG << "Add module " << v.first << " successed!";
			}
			else
			{
				WARNING << "Module type " << v.first << " is not registered!";
			}

		}
	}

	virtual ~Loop()
	{
	}
	static TR1::function<void(void)> Create(BaseContext* d, const ptree & pt)
	{
		return TR1::bind(&ThisType::Eval,
				TR1::shared_ptr<ThisType>(new ThisType(*d, pt)));
	}

	virtual void Eval()
	{

		for (size_t i = 0; i < maxstep; ++i)
		{
			std::for_each(modules.begin(), modules.end(), eval_);
		}
	}

}
;

}  // namespace flow_control

}  // namespace simpla

#endif /* FLOW_CONTROL_H_ */
