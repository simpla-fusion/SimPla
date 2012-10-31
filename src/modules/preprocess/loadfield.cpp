/*
 * loadfield.cpp
 *
 *  Created on: 2012-10-31
 *      Author: salmon
 */

#include "loadfield.h"
namespace simpla
{

namespace preprocess
{

LoadField::LoadField(BaseContext & d, const ptree & pt) :
		ctx(d), pt_(pt),

		name(pt.get<std::string>("<xmlattr>.Name")),

		type(pt.get<std::string>("<xmlattr>.Type"))
{

	if (ctx.objFactory_.find(type) == ctx.objFactory_.end())
	{
		ERROR << "Object type " << type << " is not registered!";
	}

}

LoadField::~LoadField()
{
}
 TR1::function<void(void)> LoadField::Create(BaseContext* d,
		const ptree & pt)
{
	return TR1::bind(&ThisType::Eval,
			TR1::shared_ptr<ThisType>(new ThisType(*d, pt)));
}

void LoadField::Eval()
{
	ctx.objects[name] = ctx.objFactory_[type]();

	TR1::shared_ptr<ArrayObject> obj = TR1::dynamic_pointer_cast<ArrayObject>(
			ctx.objects[name]);

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
			Complex a = pt_.get("Data", dv, pt_trans<Complex, std::string>());
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

}  // namespace preprocess

}  // namespace simpla

