/*
 * em_src.h
 *
 *  Created on: 2012-10-18
 *      Author: salmon
 */

#ifndef EM_SRC_H_
#define EM_SRC_H_

#include "include/simpla_defs.h"

#include <cmath>
#include <boost/algorithm/string.hpp>
#include <boost/optional.hpp>
#include "engine/context.h"
#include "engine/object.h"
#include "fetl/fetl.h"
#include "utilities/properties.h"

namespace simpla
{
namespace em
{

template<typename TG>
struct RFSrc
{

	DEFINE_FIELDS(typename TG::ValueType,TG)
	typedef RFSrc<TG> ThisType;
	Context<Grid> & ctx;

	RFSrc(Context<Grid> & d, const ptree & pt) :
			ctx(d),

			dt(ctx.grid.dt),

			alpha(pt.get("Arguments.alpha", 0.0f)),

			freq(pt.get("Arguments.freq", 1.0f)),

			field_name(pt.get("Data.Field", "E1")),

			field_type(pt.get("Data.Field.<xmlattr>.Type", "OneForm")),

			x(pt.get<Vec3>("Arguments.pos")),

			A(pt.get<Vec3>("Arguments.amp"))

	{
		boost::algorithm::trim(field_name);
		LOG << "Create module RFSrc";
	}

	virtual ~RFSrc()
	{
	}

	static TR1::function<void()> Create(Context<TG> * d, const ptree & pt)
	{
		return TR1::bind(&ThisType::Eval,
				TR1::shared_ptr<ThisType>(new ThisType(*d, pt)));
	}

	virtual void Eval()
	{
		LOG << "Run module RFSrc";

		Real t = ctx.Timer();

		Vec3 v = A * (1.0 - std::exp(-t * alpha)) * std::sin(freq * t);

		size_t s = ctx.grid.get_cell_num(x);

		boost::optional<TR1::shared_ptr<Object> > obj = ctx.FindObject(
				field_name);

		if (field_type == "OneForm")
		{
			ctx.grid.Scatter(*ctx.template GetObject<OneForm>(field_name), x,
					v);
		}
		else if (field_type == "TwoForm")
		{
			ctx.grid.Scatter(*ctx.template GetObject<TwoForm>(field_name), x,
					v);
		}
		else if (field_type == "VecZeroForm")
		{
			ctx.grid.Scatter(*ctx.template GetObject<VecZeroForm>(field_name),
					x, v);
		}
		else if (field_type == "COneForm")
		{
			ctx.grid.Scatter(*ctx.template GetObject<COneForm>(field_name), x,
					v);
		}
		else if (field_type == "CTwoForm")
		{
			ctx.grid.Scatter(*ctx.template GetObject<CTwoForm>(field_name), x,
					v);
		}
		else if (field_type == "CVecZeroForm")
		{
			ctx.grid.Scatter(*ctx.template GetObject<CVecZeroForm>(field_name),
					x, v);
		}

	}

private:
	const Real dt;

	const Real alpha, freq;
	const Vec3 A;
	const Vec3 x;

	std::string field_name;
	std::string field_type;

};

} //namespace ext_src
}  // namespace simpla

#endif /* EM_SRC_H_ */
