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

#include "../data_model/DataSet.h"
#include "../gtl/design_pattern/singleton_holder.h"
#include "HDF5Stream.h"

namespace simpla { namespace io
{
//void init(int argc, char **argv)
//{
////    SingletonHolder<HDF5Stream>::instance().init(argc, argv);
//}
//
//void close()
//{
////    SingletonHolder<HDF5Stream>::instance().close();
//}
//
//std::string help_message()
//{
//    return HDF5Stream::help_message();
//};
//
//std::string write(std::string const &url, data_model::DataSet const &ds, size_t flag)
//{
//    return SingletonHolder<io::HDF5Stream>::instance().write(url, ds, flag);
//}
//
//data_model::DataSet load(std::string const &url)
//{
//    data_model::DataSet ds;
//    SingletonHolder<io::HDF5Stream>::instance().read(url, &ds);
//    return std::move(ds);
//}
//
//std::string cd(std::string const &url)
//{
//    return std::get<1>(SingletonHolder<io::HDF5Stream>::instance().open(url));
//}
}//namespace io
}
// namespace simpla

