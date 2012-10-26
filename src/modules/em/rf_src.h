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

template<typename TContext>
struct RFSrc: public Module
{

	DEFINE_FIELDS(typename TContext::Grid)

TContext	& ctx;

	RFSrc(TContext & d, const ptree & pt) :
	ctx(d),

	dt(ctx.dt),

	alpha(pt.get("alpha", 0.0f)),

	freq(pt.get("freq", 1.0f)),

	field_name(pt.get<std::string>("field")),

	x(pt.get<Vec3>("pos" )),

	A(pt.get<Vec3>("amp" ))

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

		ctx.grid.Scatter(*ctx.template GetObject<OneForm>(field_name), x,
				A * (1.0 - std::exp(-t * alpha)) * std::sin(freq * t));
		CHECK(x);
		CHECK(A * (1.0 - std::exp(-t * alpha)) * std::sin(freq * t));
	}

private:
	const Real dt;

	const Real alpha, freq;
	const Vec3 A;
	const Vec3 x;

	std::string field_name;

};

} //namespace ext_src
}  // namespace simpla

#endif /* EM_SRC_H_ */
