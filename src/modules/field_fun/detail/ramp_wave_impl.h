/*
 * ramp_wave_impl.h
 *
 *  Created on: 2012-10-30
 *      Author: salmon
 */

#ifndef RAMP_WAVE_IMPL_H_
#define RAMP_WAVE_IMPL_H_

#include "field_fun_impl.h"
#include "fetl/fetl.h"
#include "fetl/grid/uniform_rect.h"
#include "engine/modules.h"
namespace simpla
{

namespace field_fun
{

template<typename TF, typename Fonctor> struct FieldFunction;

template<typename TG> struct BaseFieldFunction;

template<typename TV>
struct FieldFunction<Field<UniformRectGrid, IZeroForm, TV>, RampWave<TV> > : public BaseFieldFunction<
		UniformRectGrid>
{

	typedef TV ValueType;

	typedef RampWave<TV> FunType;

	typedef Field<UniformRectGrid, IZeroForm, TV> FieldType;

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
		LOG << "Run module RampWave";

		Real t = ctx.Timer();

		IVec3 I;

		FieldType &obj = *ctx.GetObject<FieldType>(field_name);

		for (I[0] = imin[0]; I[0] < imax[0]; ++I[0])
			for (I[1] = imin[1]; I[1] < imax[1]; ++I[1])
				for (I[2] = imin[2]; I[2] < imax[2]; ++I[2])
				{
					RVec3 x = ctx.grid.xmin + I * ctx.grid.dx;
					obj[ctx.grid.get_cell_num(I)] = fun(x, t);
				}

	}

};

template<typename TV>
struct FieldFunction<Field<UniformRectGrid, IThreeForm, TV>, RampWave<TV> > : public BaseFieldFunction<
		UniformRectGrid>
{

	typedef TV ValueType;

	typedef RampWave<TV> FunType;

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
		LOG << "Run module RampWave";

		Real t = ctx.Timer();

		IVec3 I;

		FieldType &obj = *ctx.GetObject<FieldType>(field_name);

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

template<typename TV>
struct FieldFunction<Field<UniformRectGrid, IOneForm, TV>,
		RampWave<nTuple<THREE, TV> > > : public BaseFieldFunction<
		UniformRectGrid>
{

	DEFINE_FIELDS(typename UniformRectGrid::ValueType,UniformRectGrid)

	typedef nTuple<THREE, TV> ValueType;

	typedef RampWave<ValueType> FunType;

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
		LOG << "Run module RampWave";

		Real t = ctx.Timer();

		IVec3 I;

		FieldType &obj = *ctx.GetObject<FieldType>(field_name);

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

template<typename TV>
struct FieldFunction<Field<UniformRectGrid, ITwoForm, TV>,
		RampWave<nTuple<THREE, TV> > > : public BaseFieldFunction<
		UniformRectGrid>
{

	DEFINE_FIELDS(typename UniformRectGrid::ValueType,UniformRectGrid)

	typedef nTuple<THREE, TV> ValueType;

	typedef RampWave<ValueType> FunType;

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
		LOG << "Run module RampWave";

		Real t = ctx.Timer();

		IVec3 I;

		FieldType &obj = *ctx.GetObject<FieldType>(field_name);

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

}  // namespace field_fun

}  // namespace simpla

#endif /* RAMP_WAVE_IMPL_H_ */
