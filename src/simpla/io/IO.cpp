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

#include "HDF5Stream.h"
#include "XDMFStream.h"

namespace simpla { namespace toolbox
{
std::shared_ptr<toolbox::IOStream> create_from_output_url(std::string const &url)
{
    std::shared_ptr<toolbox::IOStream> res(nullptr);

    std::string ext = url.substr(url.find_last_of('.'));

    if (ext == ".xdmf")
    {
        res.reset(new XDMFStream);
    } else
    {
        res.reset(new HDF5Stream);
    }


    res->current_file_name(url);

    return res;
}
//
//void close()
//{
//    SingletonHolder<std::unique_ptr<toolbox::IOStream>>::instance()->close();
//    SingletonHolder<std::unique_ptr<toolbox::IOStream>>::instance().reset(nullptr);
//}
//
//IOStream &global()
//{
//    return *(SingletonHolder<std::unique_ptr<toolbox::IOStream>>::instance());
//}
//
//std::string write(std::string const &url, data_model::DataSet const &ds, size_t id)
//{
//    return SingletonHolder<std::unique_ptr<toolbox::IOStream>>::instance()->write(url, ds, id);
//}
//
//data_model::DataSet Load(std::string const &url)
//{
//    data_model::DataSet ds;
//    SingletonHolder<std::unique_ptr<toolbox::IOStream>>::instance()->read(url, &ds);
//    return std::Move(ds);
//}
//
//std::string cd(std::string const &url)
//{
//    return std::Get<1>(SingletonHolder<std::unique_ptr<toolbox::IOStream>>::instance()->open(url));
//}
}}// namespace simpla//namespace io

