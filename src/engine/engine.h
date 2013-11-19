/*
 * engine.h
 *
 *  Created on: 2011-12-13
 *      Author: salmon
 */

#ifndef ENGINE_H_
#define ENGINE_H_
#include "third_part/pugixml/src/pugixml.hpp"
#include <string>
namespace simpla
{

class Engine
{
	pugi::xml_document doc_;
public:
	Engine()
	{

	}
	void ParseXML(std::string const &filename)
	{
		doc_.load_file(filename.c_str());
	}
	Object Evaluate()
	{

	}
};

}  // namespace simpla

#endif /* ENGINE_H_ */
