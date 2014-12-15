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
std::string save(std::string const & url, DataSet const & ds, size_t flag)
{
	return GLOBAL_DATA_STREAM.write(url,ds,flag);
}

std::string cd(std::string const & url)
{
	return std::get<1>(GLOBAL_DATA_STREAM.cd(url ));
}

void set_dataset_attribute(std::string const &url, DataType const & d_type,
		void const * buff)
{
	SingletonHolder<DataStream>::instance().set_attribute(url, d_type, buff);
}
void set_dataset_attribute(std::string const &url, std::string const & str)
{
	SingletonHolder<DataStream>::instance().set_attribute(url, str);
}
void get_dataset_attribute(std::string const &url, DataType const & d_type,
		void* buff)
{
	SingletonHolder<DataStream>::instance().get_attribute(url, d_type, buff);
}

void delete_dataset_attribute(std::string const &url)
{
	SingletonHolder<DataStream>::instance().delete_attribute(url);
}

bool set_dataset_attribute(std::string const &url, Properties const & prop)
{
	return SingletonHolder<DataStream>::instance().set_attribute(url, prop);
}

Properties get_dataset_attribute(std::string const &url)
{
	return std::move(SingletonHolder<DataStream>::instance().get_attribute(url));
}
}
// namespace simpla

