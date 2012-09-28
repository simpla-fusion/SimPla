/*
 * engine.h
 *
 *  Created on: 2011-12-13
 *      Author: salmon
 */

#ifndef ENGINE_H_
#define ENGINE_H_

#include "engine/context.h"
#include "engine/local_comm.h"

Context::Holder parseConfigFile(std::string const &cFile);

#endif /* ENGINE_H_ */
