/*
 * loadfield.h
 *
 *  Created on: 2012-10-31
 *      Author: salmon
 */

#ifndef LOADFIELD_H_
#define LOADFIELD_H_
#include "include/simpla_defs.h"
#include "engine/context.h"
#include "engine/modules.h"
#include "utilities/properties.h"
#include "modules/io/read_hdf5.h"
namespace simpla
{

namespace preprocess
{

class LoadField
{
public:
	typedef LoadField ThisType;

	LoadField(BaseContext & d, const ptree & pt) ;

	virtual ~LoadField();

	static TR1::function<void(void)> Create(BaseContext* d, const ptree & pt);

	virtual void Eval();

private:

	BaseContext & ctx;
	ptree pt_;
	std::string name, type;

};

}  // namespace preprocess

}  // namespace simpla

#endif /* LOADFIELD_H_ */
