/*
 * io.cpp
 *
 *  Created on: 2014年11月21日
 *      Author: salmon
 */

#include <stddef.h>
#include <string>

#include "../utilities/singleton_holder.h"
#include "data_stream.h"

namespace simpla
{

void init_io(int argc, char ** argv)
{
	SingletonHolder<DataStream>::instance().init(argc, argv);
}

void close_io()
{
	SingletonHolder<DataStream>::instance().close();
}
std::string save(std::string const & url, DataSet const & ds, size_t flag = 0UL)
{
	return GLOBAL_DATA_STREAM.write(url,ds,flag);
}

std::string cd(std::string const & url)
{
	return GLOBAL_DATA_STREAM.cd(url );
}

}
 // namespace simpla

