/*
 * field_fun_impl.h
 *
 *  Created on: 2012-10-30
 *      Author: salmon
 */

#ifndef FIELD_FUN_IMPL_H_
#define FIELD_FUN_IMPL_H_

#include "include/simpla_defs.h"
#include "utilities/properties.h"
#include "engine/context.h"
#include "engine/basecontext.h"
#include "engine/basemodule.h"

#include "fetl/fetl.h"
#include "fetl/grid/uniform_rect.h"

#include "../assign_constant.h"
#include "../smooth.h"
#include "../damping.h"
#include "../lua_script.h"
namespace simpla
{
namespace field_fun
{
template<typename TG> struct BaseFieldFunction;

template<>
struct BaseFieldFunction<UniformRectGrid> : public BaseModule
{
	DEFINE_FIELDS(typename UniformRectGrid::ValueType,UniformRectGrid)

	Context<Grid> & ctx;

	std::string field_name;

	IVec3 imin, imax;

	BaseFieldFunction(Context<Grid> * d, const ptree & pt) :
			BaseModule(d, pt), ctx(*d), field_name("default")
	{
		boost::optional<RVec3> xmin = pt.get_optional<RVec3>("Domain.XMin",
				pt_trans<RVec3, std::string>());

		boost::optional<RVec3> xmax = pt.get_optional<RVec3>("Domain.XMax",
				pt_trans<RVec3, std::string>());

		if (!!xmin && !!xmax)
		{
			imin = *xmin * ctx.grid.inv_dx;

			imax = *xmax * ctx.grid.inv_dx + 1;
		}

		else
		{
			IVec3 i0 =
			{ 0, 0, 0 };
			imin = i0;
			imax = ctx.grid.dims;
		}
	}
	~BaseFieldFunction()
	{
	}

};
template<typename, typename > struct FieldFunction;

template<typename TV, template<typename > class TFun>
struct FieldFunction<Field<UniformRectGrid, IZeroForm, TV>, TFun<TV> > : public BaseFieldFunction<
		UniformRectGrid>
{

	typedef TV ValueType;

	typedef TFun<TV> FunType;

	typedef Field<UniformRectGrid, IZeroForm, TV> FieldType;

	typedef FieldFunction<FieldType, FunType> ThisType;

	FunType fun;

	FieldFunction(Context<Grid> * d, const ptree & pt) :
			BaseFieldFunction<UniformRectGrid>(d, pt), fun(pt)
	{
		LOG << "Create module TFun: " << typeid(TFun<TV> ).name();
	}

	virtual ~FieldFunction()
	{
	}
	static TR1::function<void(void)> Create(Context<Grid> * d, const ptree & pt)
	{
		return TR1::bind(&ThisType::Eval,
				TR1::shared_ptr<ThisType>(new ThisType(d, pt)));
	}

	virtual void Eval()
	{
		LOG << "Run module Function: " << typeid(TFun<TV> ).name();

		Real t = ctx.Timer();

		IVec3 I;

		FieldType &obj = *TR1::dynamic_pointer_cast<FieldType>(
				dataset_[field_name]);

		for (I[0] = imin[0]; I[0] < imax[0]; ++I[0])
			for (I[1] = imin[1]; I[1] < imax[1]; ++I[1])
				for (I[2] = imin[2]; I[2] < imax[2]; ++I[2])
				{
					RVec3 x = ctx.grid.xmin + I * ctx.grid.dx;
					obj[ctx.grid.get_cell_num(I)] = fun(x, t);
				}

	}

};

template<typename TV, template<typename > class TFun>
struct FieldFunction<Field<UniformRectGrid, IThreeForm, TV>, TFun<TV> > : public BaseFieldFunction<
		UniformRectGrid>
{

	typedef TV ValueType;

	typedef TFun<TV> FunType;

	typedef Field<UniformRectGrid, IThreeForm, TV> FieldType;

	typedef FieldFunction<FieldType, FunType> ThisType;

	FunType fun;

	FieldFunction(Context<Grid> * d, const ptree & pt) :
			BaseFieldFunction<UniformRectGrid>(d, pt), fun(pt)
	{

	}

	virtual ~FieldFunction()
	{
	}
	static TR1::function<void(void)> Create(Context<Grid> * d, const ptree & pt)
	{
		return TR1::bind(&ThisType::Eval,
				TR1::shared_ptr<ThisType>(new ThisType(d, pt)));
	}

	virtual void Eval()
	{
		LOG << "Run module Function: " << typeid(TFun<TV> ).name();

		Real t = ctx.Timer();

		IVec3 I;

		FieldType &obj = *TR1::dynamic_pointer_cast<FieldType>(
				dataset_[field_name]);

		for (I[0] = imin[0]; I[0] < imax[0]; ++I[0])
			for (I[1] = imin[1]; I[1] < imax[1]; ++I[1])
				for (I[2] = imin[2]; I[2] < imax[2]; ++I[2])
				{
					RVec3 x = ctx.grid.xmin + I * ctx.grid.dx;
					obj[ctx.grid.get_cell_num(I)] = fun(
							x + ctx.grid.dx[0] * 0.5, t);
				}

	}

};

template<typename TV, template<typename > class TFun>
struct FieldFunction<Field<UniformRectGrid, IOneForm, TV>,
		TFun<nTuple<THREE, TV> > > : public BaseFieldFunction<UniformRectGrid>
{

	DEFINE_FIELDS(typename UniformRectGrid::ValueType,UniformRectGrid)

	typedef nTuple<THREE, TV> ValueType;

	typedef TFun<ValueType> FunType;

	typedef Field<UniformRectGrid, IOneForm, TV> FieldType;

	typedef FieldFunction<FieldType, FunType> ThisType;

	FunType fun;

	FieldFunction(Context<Grid> *d, const ptree & pt) :
			BaseFieldFunction<UniformRectGrid>(d, pt), fun(pt)
	{
	}

	virtual ~FieldFunction()
	{
	}
	static TR1::function<void(void)> Create(Context<Grid> * d, const ptree & pt)
	{
		return TR1::bind(&ThisType::Eval,
				TR1::shared_ptr<ThisType>(new ThisType(d, pt)));
	}
	virtual void Eval()
	{
		LOG << "Run module Function: " << typeid(TFun<TV> ).name();

		Real t = ctx.Timer();

		IVec3 I;

		FieldType &obj = *TR1::dynamic_pointer_cast<FieldType>(
				dataset_[field_name]);

		RVec3 DX =
		{ ctx.grid.dx[0] * 0.5, 0, 0 };
		RVec3 DY =
		{ 0, ctx.grid.dx[0] * 0.5, 0 };
		RVec3 DZ =
		{ 0, 0, ctx.grid.dx[0] * 0.5 };

		for (I[0] = imin[0]; I[0] < imax[0]; ++I[0])
			for (I[1] = imin[1]; I[1] < imax[1]; ++I[1])
				for (I[2] = imin[2]; I[2] < imax[2]; ++I[2])
				{
					RVec3 x = ctx.grid.xmin + I * ctx.grid.dx;

					size_t s = ctx.grid.get_cell_num(I);

					obj[s * 3 + 0] = fun(x + DX, t)[0];
					obj[s * 3 + 1] = fun(x + DY, t)[1];
					obj[s * 3 + 2] = fun(x + DZ, t)[2];
				}

	}

};

template<typename TV, template<typename > class TFun>
struct FieldFunction<Field<UniformRectGrid, ITwoForm, TV>,
		TFun<nTuple<THREE, TV> > > : public BaseFieldFunction<UniformRectGrid>
{

	DEFINE_FIELDS(typename UniformRectGrid::ValueType,UniformRectGrid)

	typedef nTuple<THREE, TV> ValueType;

	typedef TFun<ValueType> FunType;

	typedef Field<UniformRectGrid, ITwoForm, TV> FieldType;

	typedef FieldFunction<FieldType, FunType> ThisType;

	FunType fun;

	FieldFunction(Context<Grid> * d, const ptree & pt) :
			BaseFieldFunction<UniformRectGrid>(d, pt), fun(pt)
	{
	}

	virtual ~FieldFunction()
	{

	}
	static TR1::function<void(void)> Create(Context<Grid> * d, const ptree & pt)
	{
		return TR1::bind(&ThisType::Eval,
				TR1::shared_ptr<ThisType>(new ThisType(d, pt)));
	}
	virtual void Eval()
	{
		LOG << "Run module Function: " << typeid(TFun<TV> ).name();

		Real t = ctx.Timer();

		IVec3 I;

		FieldType &obj = *TR1::dynamic_pointer_cast<FieldType>(
				dataset_[field_name]);

		RVec3 DX =
		{ ctx.grid.dx[0] * 0.5, 0, 0 };
		RVec3 DY =
		{ 0, ctx.grid.dx[0] * 0.5, 0 };
		RVec3 DZ =
		{ 0, 0, ctx.grid.dx[0] * 0.5 };

		for (I[0] = imin[0]; I[0] < imax[0]; ++I[0])
			for (I[1] = imin[1]; I[1] < imax[1]; ++I[1])
				for (I[2] = imin[2]; I[2] < imax[2]; ++I[2])
				{
					RVec3 x = ctx.grid.xmin + I * ctx.grid.dx;
					size_t s = ctx.grid.get_cell_num(I);

					obj[s * 3 + 0] = fun(x + DY + DZ, t)[0];
					obj[s * 3 + 1] = fun(x + DZ + DX, t)[1];
					obj[s * 3 + 2] = fun(x + DX + DY, t)[2];
				}

	}

};
template<typename TG, template<typename > class TFun> inline TR1::function<
		void(void)> Create(Context<TG>* ctx, ptree const & pt)
{

	boost::optional<std::string> field_type = pt.get<std::string>(
			"DataSet.<xmlattr>.Type");

	TR1::function<void(void)> res;

	//FIXME object should not have been defined at here!!!

	if (!field_type)
	{
		ERROR << "Unknown field type " << field_type << "!";
	}
	//Real
	else if (*field_type == "ZeroForm")
	{
		res = FieldFunction<Field<TG, IZeroForm, Real>, TFun<Real> >::Create(
				ctx, pt);
	}
	else if (*field_type == "OneForm")
	{
		res = FieldFunction<Field<TG, IOneForm, Real>,
				TFun<nTuple<THREE, Real> > >::Create(ctx, pt);
	}
	else if (*field_type == "TwoForm")
	{
		res = FieldFunction<Field<TG, ITwoForm, Real>,
				TFun<nTuple<THREE, Real> > >::Create(ctx, pt);
	}
	else if (*field_type == "ThreeForm")
	{
		res = FieldFunction<Field<TG, IThreeForm, Real>, TFun<Real> >::Create(
				ctx, pt);
	}
	// Vec3
	else if (*field_type == "ZeroForm")
	{
		res = FieldFunction<Field<TG, IZeroForm, Vec3>, TFun<Vec3> >::Create(
				ctx, pt);
	}
	else if (*field_type == "OneForm")
	{
		res = FieldFunction<Field<TG, IOneForm, Vec3>,
				TFun<nTuple<THREE, Vec3> > >::Create(ctx, pt);
	}
	else if (*field_type == "TwoForm")
	{
		res = FieldFunction<Field<TG, ITwoForm, Vec3>,
				TFun<nTuple<THREE, Vec3> > >::Create(ctx, pt);
	}
	else if (*field_type == "ThreeForm")
	{
		res = FieldFunction<Field<TG, IThreeForm, Vec3>, TFun<Vec3> >::Create(
				ctx, pt);
	}
	// Complex
	else if (*field_type == "ZeroForm")
	{
		res =
				FieldFunction<Field<TG, IZeroForm, Complex>, TFun<Complex> >::Create(
						ctx, pt);
	}
	else if (*field_type == "OneForm")
	{
		res = FieldFunction<Field<TG, IOneForm, Complex>,
				TFun<nTuple<THREE, Complex> > >::Create(ctx, pt);
	}
	else if (*field_type == "TwoForm")
	{
		res = FieldFunction<Field<TG, ITwoForm, Complex>,
				TFun<nTuple<THREE, Complex> > >::Create(ctx, pt);
	}
	else if (*field_type == "ThreeForm")
	{
		res =
				FieldFunction<Field<TG, IThreeForm, Complex>, TFun<Complex> >::Create(
						ctx, pt);
	}
	// Complex Vec3
	else if (*field_type == "ZeroForm")
	{
		res = FieldFunction<Field<TG, IZeroForm, CVec3>, TFun<CVec3> >::Create(
				ctx, pt);
	}
	else if (*field_type == "OneForm")
	{
		res = FieldFunction<Field<TG, IOneForm, CVec3>,
				TFun<nTuple<THREE, CVec3> > >::Create(ctx, pt);
	}
	else if (*field_type == "TwoForm")
	{
		res = FieldFunction<Field<TG, ITwoForm, CVec3>,
				TFun<nTuple<THREE, CVec3> > >::Create(ctx, pt);
	}
	else if (*field_type == "ThreeForm")
	{
		res = FieldFunction<Field<TG, IThreeForm, CVec3>, TFun<CVec3> >::Create(
				ctx, pt);
	}

	return res;
}

template<typename TG> inline void RegisterModules(Context<TG> * ctx)
{
	DEFINE_FIELDS(typename TG::ValueType, TG)

	ctx->moduleFactory_["AssignConstant"] = TR1::bind(
			&Create<TG, AssignConstant>, ctx, TR1::placeholders::_1);

	ctx->moduleFactory_["LuaScript"] = TR1::bind(&Create<TG, LuaScript>, ctx,
			TR1::placeholders::_1);

//	moduleFactory_["Smooth"] = TR1::bind(
//			&field_fun::Create<TG, field_fun::Smooth>, ctx,
//			TR1::placeholders::_1);
//
//	moduleFactory_["Damping"] = TR1::bind(
//			&field_fun::Create<TG, field_fun::Damping>, ctx,
//			TR1::placeholders::_1);

}

}  // namespace field_op

}  // namespace simpla

#endif /* FIELD_FUN_IMPL_H_ */
