/*Copyright (C) 2007-2011 YU Zhi. All rights reserved.
 *
 * $Id$
 * Engine/Context.h
 *
 *  Created on: 2010-11-10
 *      Author: salmon
 */

#ifndef CONTEXT_H_
#define CONTEXT_H_
#include "domain.h"

namespace simpla
{

class Modules
{
public:
	typedef Modules ThisType;

	typedef TR1::shared_ptr<ThisType> Holder;

	Domain & domain;

	//TODO input dataflow
	//TODO output dataflow

	Modules(Domain & d) :
			domain(d)
	{
	}

	virtual ~Modules()
	{
	}

	virtual void Eval()=0;
private:

	Modules(ThisType const &);
	Modules & operator=(ThisType const &);

}
;

}  // namespace simpla
#endif   // CONTEXT_H_
