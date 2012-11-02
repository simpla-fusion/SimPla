/*
 * preprocess.cpp
 *
 *  Created on: 2012-11-2
 *      Author: salmon
 */
#include "include/simpla_defs.h"
#include "preprocess.h"

#include "datastruct/ndarray.h"
#include "datastruct/pool.h"
#include "datastruct/compound.h"

#include "modules/io/read_hdf5.h"
#include "utilities/properties.h"
#include "fetl/ntuple.h"

namespace simpla
{
namespace preprocess
{
void Preprocess(BaseContext * ctx, const ptree & pt)
{
	BOOST_FOREACH(const typename ptree::value_type &v, pt)
	{
		boost::optional<std::string> name = v.second.get<std::string>(
				"<xmlattr>.Name");

		if (!!name)
		{
			if (v.first == "Field")
			{
				ctx->objects[*name] = LoadField(ctx, v.second);
			}
			else if (v.first == "ParticlePool")
			{
				ctx->objects[*name] = LoadPool(ctx, v.second);
			}
			else if (v.first == "Compound")
			{
				ctx->objects[*name] = LoadCompound(ctx, v.second);
			}

			LOG << "Load " << v.first << " object [" << *name << "].";
		}
	}
}

TR1::shared_ptr<NdArray> LoadField(BaseContext * ctx, const ptree & pt)
{

	std::string type = pt.get<std::string>("<xmlattr>.Type");

	if (ctx->objFactory_.find(type) == ctx->objFactory_.end())
	{
		ERROR << "Unknown object type " << type;
	}

	TR1::shared_ptr<NdArray> res = TR1::dynamic_pointer_cast<NdArray>(
			ctx->objFactory_[type]());

	boost::optional<std::string> format;

	format = pt.get_optional<std::string>("Value.<xmlattr>.Format");

	if (!format)
	{
		res->Clear();
	}
	else if (*format == "HDF")
	{
		std::string url = pt.get_value<std::string>();
		io::ReadData(url, res);
	}
	else if (*format == "XML")
	{
		if (res->CheckValueType(typeid(Integral)))
		{
			res->FullFill(pt.get_value(0));
		}
		else if (res->CheckValueType(typeid(Real)))
		{
			res->FullFill(pt.get_value(0.0d));
		}
		else if (res->CheckValueType(typeid(Complex)))
		{
			Complex dv(0, 0);
			Complex a = pt.get_value(dv, pt_trans<Complex, std::string>());
			res->FullFill(a);
		}
		else if (res->CheckValueType(typeid(nTuple<THREE, Real> )))
		{
			nTuple<THREE, Real> dv =
			{ 0, 0, 0 };
			nTuple<THREE, Real> a = pt.get_value(dv,
					pt_trans<nTuple<THREE, Real>, std::string>());
			res->FullFill(a);
		}
		else if (res->CheckValueType(typeid(nTuple<THREE, Complex> )))
		{
			nTuple<THREE, Complex> dv =
			{ 0, 0, 0 };
			nTuple<THREE, Complex> a = pt.get_value(dv,
					pt_trans<nTuple<THREE, Complex>, std::string>());
			res->FullFill(a);
		}

	}

	return res;

}

TR1::shared_ptr<Pool> LoadPool(BaseContext * ctx, const ptree & pt)
{
	std::string type = pt.get<std::string>("<xmlattr>.Type");

	if (ctx->objFactory_.find(type) == ctx->objFactory_.end())
	{
		ERROR << "Unknown object type " << type;
	}

	TR1::shared_ptr<Pool> res = TR1::dynamic_pointer_cast<Pool>(
			ctx->objFactory_[type]());

	return res;
}
TR1::shared_ptr<CompoundObject> LoadCompound(BaseContext * ctx,
		const ptree & pt)
{
	TR1::shared_ptr<CompoundObject> obj(new CompoundObject);

	BOOST_FOREACH(const typename ptree::value_type &v, pt)
	{

		if (v.first == "Field")
		{
			obj->objects[v.second.get<std::string>("<xmlattr>.Name")] =
					LoadField(ctx, v.second);
		}
		else if (v.first == "Pool")
		{
			obj->objects[v.second.get<std::string>("<xmlattr>.Name")] =
					LoadPool(ctx, v.second);
		}
		else if (v.first == "Compound")
		{
			obj->objects[v.second.get<std::string>("<xmlattr>.Name")] =
					LoadCompound(ctx, v.second);
		}
		else if (v.first == "<xmlattr>")
		{
			obj->properties = v.second;
		}

	}
	return obj;
}

}  // namespace preprocess

}  // namespace simpla
