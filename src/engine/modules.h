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

namespace simpla
{

class Modules
{
public:
	typedef Modules ThisType;

	typedef TR1::shared_ptr<ThisType> Holder;

	//TODO input dataflow
	//TODO output dataflow

	Modules()
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
