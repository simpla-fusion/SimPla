/**
 * @file io.cpp
 *
 *  Created on: 2014-11-21
 *      @Author: salmon
 */

#include <stddef.h>
#include <algorithm>
#include <string>
#include <tuple>

#include "../gtl/dataset/dataset.h"
#include "../gtl/design_pattern/singleton_holder.h"
#include "data_stream.h"

namespace simpla
{

std::string init_io(int argc, char ** argv)
{
	return SingletonHolder<DataStream>::instance().init(argc, argv);
}

void close_io()
{
	SingletonHolder<DataStream>::instance().close();
}
std::string save(std::string const & url, DataSet const & ds, size_t flag)
{
	return GLOBAL_DATA_STREAM.write(url,ds,flag);
}

DataSet load(std::string const & url)
{
	DataSet ds;
	GLOBAL_DATA_STREAM.read(url,&ds );
	return std::move(ds);
}

std::string cd(std::string const & url)
{
	return std::get<1>(GLOBAL_DATA_STREAM.cd(url ));
}

}
// namespace simpla

