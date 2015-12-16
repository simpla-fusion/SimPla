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

#include "../data_model/dataset.h"
#include "../gtl/design_pattern/singleton_holder.h"
#include "data_stream.h"

namespace simpla { namespace io
{
void init(int argc, char **argv)
{
    SingletonHolder<DataStream>::instance().init(argc, argv);
}

void close()
{
    SingletonHolder<DataStream>::instance().close();
}

std::string help_message()
{
    return DataStream::help_message();
};

std::string save(std::string const &url, DataSet const &ds, size_t flag)
{
    return SingletonHolder<io::DataStream>::instance().write(url, ds, flag);
}

DataSet load(std::string const &url)
{
    DataSet ds;
    SingletonHolder<io::DataStream>::instance().read(url, &ds);
    return std::move(ds);
}

std::string cd(std::string const &url)
{
    return std::get<1>(SingletonHolder<io::DataStream>::instance().cd(url));
}
}//namespace io
}
// namespace simpla

