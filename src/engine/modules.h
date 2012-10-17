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

class Module
{
public:
	typedef Module ThisType;

	typedef TR1::shared_ptr<ThisType> Holder;

	//TODO input dataflow
	//TODO output dataflow

	Module()
	{
	}

	virtual ~Module()
	{
	}

	virtual void Eval()=0;
private:

	Module(ThisType const &);
	Module & operator=(ThisType const &);

}
;

}  // namespace simpla
#endif   // CONTEXT_H_
