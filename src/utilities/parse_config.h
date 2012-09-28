/*
 * parse_config.h
 *
 *  Created on: 2012-3-6
 *      Author: salmon
 */

#ifndef PARSE_CONFIG_H_
#define PARSE_CONFIG_H_
#include "data_struct/properties.h"
#include <string>

Properties ParseConfig(int argc, char** argv);
Properties ParseConfigFile(std::string const &);
#endif /* PARSE_CONFIG_H_ */
