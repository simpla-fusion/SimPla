/*
 * Solver.h
 *
 *  Created on: 2012-3-19
 *      Author: salmon
 */

#ifndef SOLVER_H_
#define SOLVER_H_
#include "include/simpla_defs.h"
namespace simpla
{

class Solver
{
public:
	typedef Solver ThisType;
	typedef TR1::shared_ptr<ThisType> Holder;

	Solver();

	virtual ~Solver()=0;

	virtual void PreProcess()=0;
	virtual void Process()=0;
	virtual void PostProcess()=0;

private:
	const std::string name_;
};

Solver::Solver()
{
}
Solver::~Solver()
{
}
void Solver::PreProcess()
{
}
void Solver::Process()
{
}
void Solver::PostProcess()
{
}

} // namespace simpla

#endif /* SOLVER_H_ */
