/*
 * loadparticle.h
 *
 *  Created on: 2012-10-31
 *      Author: salmon
 */

#ifndef LOADPARTICLE_H_
#define LOADPARTICLE_H_

#include "include/simpla_defs.h"
#include "engine/context.h"
#include "utilities/properties.h"
#include "modules/io/read_hdf5.h"
namespace simpla
{

namespace preprocess
{

class LoadParticle
{
public:
	typedef LoadParticle ThisType;

	LoadParticle(BaseContext & d, const ptree & pt);

	virtual ~LoadParticle();

	static TR1::function<void(void)> Create(BaseContext* d, const ptree & pt);

	virtual void Eval();

private:

	BaseContext & ctx;
	ptree pt_;
	std::string name, type;

};

}  // namespace preprocess

}  // namespace simpla

#endif /* LOADPARTICLE_H_ */
