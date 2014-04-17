/*Copyright (C) 2007-2011 YU Zhi. All rights reserved.
 *
 *  Created on: 2011-3-22
 *      Author: salmon
 *
 *  local_comm.h
 * 
 */

#ifndef SRC_ENGINE_LOCAL_COMM_H_
#define SRC_ENGINE_LOCAL_COMM_H_
#include <utility>
#include <list>
#include <string>

#include "engine/context.h"
#include "fetl/ntuple.h"

class Context;

class LocalComm {
public:
	typedef LocalComm ThisType;
	typedef TR1::shared_ptr<ThisType> Holder;

	explicit LocalComm(Context::Holder ctx);

	inline static Holder create(Context::Holder ctx) {
		return (Holder(new ThisType(ctx)));
	}

	virtual ~LocalComm() {
	}
	void updateField(std::string const &name);
	void updateParticle(std::string const &name) {
		WARNING("!!!UNIMPLEMENT!!!")
	}
private:
	Context::Holder ctx_;
	std::list<IVec3> neighour;
};

#endif  // SRC_ENGINE_LOCAL_COMM_H_
