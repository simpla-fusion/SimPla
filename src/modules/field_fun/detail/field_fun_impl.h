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

	boost::optional<TR1::shared_ptr<Object> > obj = ctx->FindObject(field_name);

	TR1::function<void(void)> res;


	//FIXME object should not have been defined at here!!!

	if (!obj)
	{
		ERROR << "Field " << field_name << " is not defined!!";
	}
	//Real
	else if ((*obj)->CheckType(typeid(Field<TG, IZeroForm, Real> )))
	{
		res = FieldFunction<Field<TG, IZeroForm, Real>, TFun<Real> >::Create(
				ctx, pt);
	}
	else if ((*obj)->CheckType(typeid(Field<TG, IOneForm, Real> )))
	{
		res = FieldFunction<Field<TG, IOneForm, Real>,
				TFun<nTuple<THREE, Real> > >::Create(ctx, pt);
	}
	else if ((*obj)->CheckType(typeid(Field<TG, ITwoForm, Real> )))
	{
		res = FieldFunction<Field<TG, ITwoForm, Real>,
				TFun<nTuple<THREE, Real> > >::Create(ctx, pt);
	}
	else if ((*obj)->CheckType(typeid(Field<TG, IThreeForm, Real> )))
	{
		res = FieldFunction<Field<TG, IThreeForm, Real>, TFun<Real> >::Create(
				ctx, pt);
	}
	// Vec3
	else if ((*obj)->CheckType(typeid(Field<TG, IZeroForm, Vec3> )))
	{
		res = FieldFunction<Field<TG, IZeroForm, Vec3>, TFun<Vec3> >::Create(
				ctx, pt);
	}
	else if ((*obj)->CheckType(typeid(Field<TG, IOneForm, Vec3> )))
	{
		res = FieldFunction<Field<TG, IOneForm, Vec3>,
				TFun<nTuple<THREE, Vec3> > >::Create(ctx, pt);
	}
	else if ((*obj)->CheckType(typeid(Field<TG, ITwoForm, Vec3> )))
	{
		res = FieldFunction<Field<TG, ITwoForm, Vec3>,
				TFun<nTuple<THREE, Vec3> > >::Create(ctx, pt);
	}
	else if ((*obj)->CheckType(typeid(Field<TG, IThreeForm, Vec3> )))
	{
		res = FieldFunction<Field<TG, IThreeForm, Vec3>, TFun<Vec3> >::Create(
				ctx, pt);
	}
	// Complex
	else if ((*obj)->CheckType(typeid(Field<TG, IZeroForm, Complex> )))
	{
		res =
				FieldFunction<Field<TG, IZeroForm, Complex>, TFun<Complex> >::Create(
						ctx, pt);
	}
	else if ((*obj)->CheckType(typeid(Field<TG, IOneForm, Complex> )))
	{
		res = FieldFunction<Field<TG, IOneForm, Complex>,
				TFun<nTuple<THREE, Complex> > >::Create(ctx, pt);
	}
	else if ((*obj)->CheckType(typeid(Field<TG, ITwoForm, Complex> )))
	{
		res = FieldFunction<Field<TG, ITwoForm, Complex>,
				TFun<nTuple<THREE, Complex> > >::Create(ctx, pt);
	}
	else if ((*obj)->CheckType(typeid(Field<TG, IThreeForm, Complex> )))
	{
		res =
				FieldFunction<Field<TG, IThreeForm, Complex>, TFun<Complex> >::Create(
						ctx, pt);
	}
	// Complex Vec3
	else if ((*obj)->CheckType(typeid(Field<TG, IZeroForm, CVec3> )))
	{
		res = FieldFunction<Field<TG, IZeroForm, CVec3>, TFun<CVec3> >::Create(
				ctx, pt);
	}
	else if ((*obj)->CheckType(typeid(Field<TG, IOneForm, CVec3> )))
	{
		res = FieldFunction<Field<TG, IOneForm, CVec3>,
				TFun<nTuple<THREE, CVec3> > >::Create(ctx, pt);
	}
	else if ((*obj)->CheckType(typeid(Field<TG, ITwoForm, CVec3> )))
	{
		res = FieldFunction<Field<TG, ITwoForm, CVec3>,
				TFun<nTuple<THREE, CVec3> > >::Create(ctx, pt);
	}
	else if ((*obj)->CheckType(typeid(Field<TG, IThreeForm, CVec3> )))
	{
		res = FieldFunction<Field<TG, IThreeForm, CVec3>, TFun<CVec3> >::Create(
				ctx, pt);
	}

	return res;
}

template<>
struct BaseFieldFunction<UniformRectGrid> : public Module
{
	DEFINE_FIELDS(typename UniformRectGrid::ValueType,UniformRectGrid)

	Context<Grid> & ctx;

	std::string field_name;

	IVec3 imin, imax;

	BaseFieldFunction(Context<Grid> & d, const ptree & pt) :
			ctx(d),

			field_name(pt.get<std::string>("Data.Field"))
	{
		RVec3 xmin = pt.get<RVec3>("Domain.XMin",
				pt_trans<RVec3, std::string>());

		RVec3 xmax = pt.get<RVec3>("Domain.XMax",
				pt_trans<RVec3, std::string>());

		imin = xmin * ctx.grid.inv_dx;

		imax = xmax * ctx.grid.inv_dx + 1;

		LOG << "Create module Function";
	}
	~BaseFieldFunction()
	{

	}

};

}  // namespace field_op

}  // namespace simpla

#endif /* FIELD_FUN_IMPL_H_ */
