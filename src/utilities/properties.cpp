/*
 * properties.cpp
 *
 *  Created on: 2012-10-7
 *      Author: salmon
 */

#include "properties.h"
namespace simpla
{

void read_file(std::string const & fname, ptree & pt)
{
	std::cout << fname << std::endl;

	std::string ext = fname.substr(fname.rfind(".")+1);

	if (ext == "xml")
	{
		read_xml(fname, pt);
	}
	else if (ext == "ini")
	{
		read_ini(fname, pt);
	}
	else if (ext == "json")
	{
		read_json(fname, pt);
	}
	else if (ext == "info")
	{
		read_info(fname, pt);
	}
}

void write_file(std::string const & fname, ptree const & pt)
{
	std::string ext = fname.substr(fname.rfind(".")+1);

	if (ext == "xml")
	{
		write_xml(fname, pt);
	}
	else if (ext == "ini")
	{
		write_ini(fname, pt);
	}
	else if (ext == "json")
	{
		write_json(fname, pt);
	}
	else if (ext == "info")
	{
		write_info(fname, pt);
	}
}
} //namespace simpla
