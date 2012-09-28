/*Copyright (C) 2007-2011 YU Zhi. All rights reserved.
 *
 * $Id$
 * MHDOhmLaw.h
 *
 *  Created on: 2010-12-7
 *      Author: salmon
 */

#ifndef SRC_FLUID_OHM_LAW_H_
#define SRC_FLUID_OHM_LAW_H_
#include <string>
#include <map>
#include "../engine/context.h"

namespace Fluid {

class OhmLaw {
public:

	typedef OhmLaw ThisType;

	typedef TR1::shared_ptr<ThisType> Holder;

	explicit OhmLaw(Context::Holder pctx);

	~OhmLaw();
	inline static Holder create(Context::Holder ctx) {
		return (Holder(new ThisType(ctx)));
	}

	void pre_process(std::list<std::string> const & splist);

	void post_process() {
	}

	void process();

	Context::Holder ctx;

	Context::VecZeroForm::Holder Ev;

	std::list<std::string> splist_;
};
} // namespace Fluid

#endif  // SRC_FLUID_OHM_LAW_H_
