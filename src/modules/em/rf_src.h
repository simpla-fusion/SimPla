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

#include "engine/context.h"
#include "engine/modules.h"
#include "fetl/fetl.h"
#include "utilities/properties.h"

namespace simpla
{
namespace em
{

template<typename TG>
struct RFSrc: public Module
{

	DEFINE_FIELDS(typename TG::ValueType,TG)

	Context<Grid> & ctx;

	RFSrc(Context<Grid> & d, const ptree & pt) :
			ctx(d),

			dt(ctx.dt),

			alpha(pt.get("alpha", 0.0f)),

			freq(pt.get("freq", 1.0f)),

			field_name(pt.get("Parameters.field", "E1")),

			field_type(pt.get("Parameters.field.<xmlattr>.type", "OneForm")),

			x(pt.get<Vec3>("pos")),

			A(pt.get<Vec3>("amp"))

	{
		boost::algorithm::trim(field_name);
		LOG << "Create module RFSrc";
	}

	virtual ~RFSrc()
	{
	}

	virtual void Eval()
	{
		LOG << "Run module RFSrc";

		Real t = ctx.Timer();

		Vec3 v = A * (1.0 - std::exp(-t * alpha)) * std::sin(freq * t);

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
