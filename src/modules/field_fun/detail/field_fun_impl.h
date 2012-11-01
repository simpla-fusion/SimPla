/*
 * field_fun_impl.h
 *
 *  Created on: 2012-10-30
 *      Author: salmon
 */

#ifndef FIELD_FUN_IMPL_H_
#define FIELD_FUN_IMPL_H_
#include "fetl/grid/uniform_rect.h"
#include "engine/basecontext.h"
#include "fetl/grid/uniform_rect.h"

namespace simpla
{
namespace field_fun
{
template<typename TF, typename Fonctor> struct FieldFunction;

template<typename TG> struct BaseFieldFunction;

template<typename TG, template<typename > class TFun>
TR1::function<void(void)> Create(Context<TG>* ctx, ptree const & pt)
{
	std::string field_name = pt.get<std::string>("Data.Field");
	std::string field_type = pt.get<std::string>("Data.Field.<xmlattr>.Type");

	TR1::function<void(void)> res;

	//FIXME object should not have been defined at here!!!

	if (false)
	{
		ERROR << "Field " << field_name << " is not defined!!";
	}
	//Real
	else if (field_type == "ZeroForm")
	{
		res = FieldFunction<Field<TG, IZeroForm, Real>, TFun<Real> >::Create(
				ctx, pt);
	}
	else if (field_type == "OneForm")
	{
		res = FieldFunction<Field<TG, IOneForm, Real>,
				TFun<nTuple<THREE, Real> > >::Create(ctx, pt);
	}
	else if (field_type == "TwoForm")
	{
		res = FieldFunction<Field<TG, ITwoForm, Real>,
				TFun<nTuple<THREE, Real> > >::Create(ctx, pt);
	}
	else if (field_type == "ThreeForm")
	{
		res = FieldFunction<Field<TG, IThreeForm, Real>, TFun<Real> >::Create(
				ctx, pt);
	}
	// Vec3
	else if (field_type == "ZeroForm")
	{
		res = FieldFunction<Field<TG, IZeroForm, Vec3>, TFun<Vec3> >::Create(
				ctx, pt);
	}
	else if (field_type == "OneForm")
	{
		res = FieldFunction<Field<TG, IOneForm, Vec3>,
				TFun<nTuple<THREE, Vec3> > >::Create(ctx, pt);
	}
	else if (field_type == "TwoForm")
	{
		res = FieldFunction<Field<TG, ITwoForm, Vec3>,
				TFun<nTuple<THREE, Vec3> > >::Create(ctx, pt);
	}
	else if (field_type == "ThreeForm")
	{
		res = FieldFunction<Field<TG, IThreeForm, Vec3>, TFun<Vec3> >::Create(
				ctx, pt);
	}
	// Complex
	else if (field_type == "ZeroForm")
	{
		res =
				FieldFunction<Field<TG, IZeroForm, Complex>, TFun<Complex> >::Create(
						ctx, pt);
	}
	else if (field_type == "OneForm")
	{
		res = FieldFunction<Field<TG, IOneForm, Complex>,
				TFun<nTuple<THREE, Complex> > >::Create(ctx, pt);
	}
	else if (field_type == "TwoForm")
	{
		res = FieldFunction<Field<TG, ITwoForm, Complex>,
				TFun<nTuple<THREE, Complex> > >::Create(ctx, pt);
	}
	else if (field_type == "ThreeForm")
	{
		res =
				FieldFunction<Field<TG, IThreeForm, Complex>, TFun<Complex> >::Create(
						ctx, pt);
	}
	// Complex Vec3
	else if (field_type == "ZeroForm")
	{
		res = FieldFunction<Field<TG, IZeroForm, CVec3>, TFun<CVec3> >::Create(
				ctx, pt);
	}
	else if (field_type == "OneForm")
	{
		res = FieldFunction<Field<TG, IOneForm, CVec3>,
				TFun<nTuple<THREE, CVec3> > >::Create(ctx, pt);
	}
	else if (field_type == "TwoForm")
	{
		res = FieldFunction<Field<TG, ITwoForm, CVec3>,
				TFun<nTuple<THREE, CVec3> > >::Create(ctx, pt);
	}
	else if (field_type == "ThreeForm")
	{
		res = FieldFunction<Field<TG, IThreeForm, CVec3>, TFun<CVec3> >::Create(
				ctx, pt);
	}

	return res;
}

template<>
struct BaseFieldFunction<UniformRectGrid>
{
	DEFINE_FIELDS(typename UniformRectGrid::ValueType,UniformRectGrid)

	Context<Grid> & ctx;

	std::string field_name;

	IVec3 imin, imax;

	BaseFieldFunction(Context<Grid> & d, const ptree & pt) :
			ctx(d),

			field_name(pt.get<std::string>("Data.Field"))
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
		LOG << "Create module Function";
	}
	~BaseFieldFunction()
	{

	}

};

template<typename TF, typename Fonctor> struct FieldFunction;

template<typename TG> struct BaseFieldFunction;

template<typename TV, template<typename > class TFun>
struct FieldFunction<Field<UniformRectGrid, IZeroForm, TV>, TFun<TV> > : public BaseFieldFunction<
		UniformRectGrid>
{

	typedef TV ValueType;

	typedef TFun<TV> FunType;

	typedef Field<UniformRectGrid, IZeroForm, TV> FieldType;

	typedef FieldFunction<FieldType, FunType> ThisType;

	FunType fun;

	FieldFunction(Context<Grid> & d, const ptree & pt) :
			BaseFieldFunction<UniformRectGrid>(d, pt), fun(
					pt.get_child("Arguments"))
	{

	}

	virtual ~FieldFunction()
	{
	}
	static TR1::function<void(void)> Create(Context<Grid> * d, const ptree & pt)
	{
		return TR1::bind(&ThisType::Eval,
				TR1::shared_ptr<ThisType>(new ThisType(*d, pt)));
	}

	virtual void Eval()
	{
		LOG << "Run module TFun";

		Real t = ctx.Timer();

		IVec3 I;

		FieldType &obj = *ctx.GetObject < FieldType > (field_name);

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

	FieldFunction(Context<Grid> & d, const ptree & pt) :
			BaseFieldFunction<UniformRectGrid>(d, pt),

			fun(pt.get_child("Arguments"))
	{

	}

	virtual ~FieldFunction()
	{
	}
	static TR1::function<void(void)> Create(Context<Grid> * d, const ptree & pt)
	{
		return TR1::bind(&ThisType::Eval,
				TR1::shared_ptr<ThisType>(new ThisType(*d, pt)));
	}

	virtual void Eval()
	{
		LOG << "Run module TFun";

		Real t = ctx.Timer();

		IVec3 I;

		FieldType &obj = *ctx.GetObject < FieldType > (field_name);

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

	FieldFunction(Context<Grid> & d, const ptree & pt) :
			BaseFieldFunction<UniformRectGrid>(d, pt), fun(
					pt.get_child("Arguments"))

	{
	}

	virtual ~FieldFunction()
	{
	}
	static TR1::function<void(void)> Create(Context<Grid> * d, const ptree & pt)
	{
		return TR1::bind(&ThisType::Eval,
				TR1::shared_ptr<ThisType>(new ThisType(*d, pt)));
	}
	virtual void Eval()
	{
		LOG << "Run module TFun";

		Real t = ctx.Timer();

		IVec3 I;

		FieldType &obj = *ctx.GetObject < FieldType > (field_name);

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

	FieldFunction(Context<Grid> & d, const ptree & pt) :
			BaseFieldFunction<UniformRectGrid>(d, pt), fun(
					pt.get_child("Arguments"))

	{
	}

	virtual ~FieldFunction()
	{

	}
	static TR1::function<void(void)> Create(Context<Grid> * d, const ptree & pt)
	{
		return TR1::bind(&ThisType::Eval,
				TR1::shared_ptr<ThisType>(new ThisType(*d, pt)));
	}
	virtual void Eval()
	{
		LOG << "Run module TFun";

		Real t = ctx.Timer();

		IVec3 I;

		FieldType &obj = *ctx.GetObject < FieldType > (field_name);

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

}  // namespace field_op

}  // namespace simpla

#endif /* FIELD_FUN_IMPL_H_ */
