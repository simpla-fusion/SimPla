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
#include "../gtl/design_pattern/SingletonHolder.h"
#include "../gtl/parse_command_line.h"
#include "HDF5Stream.h"
#include "XDMFStream.h"

namespace simpla { namespace io
{
std::shared_ptr<io::IOStream> create_from_args(int argc, char **argv)
{
    std::shared_ptr<io::IOStream> res(nullptr);

    std::string url = "untitled.h5";
    parse_cmd_line(
            argc, argv,
            [&](std::string const &opt, std::string const &value) -> int
            {
                if (opt == "o")
                {
                    url = value;

                    return TERMINATE;
                }
                return CONTINUE;
            }

    );
    std::string ext = url.substr(url.find_last_of('.'));

    if (ext == ".xdmf")
    {
        res.reset(new XDMFStream);
    }
    else
    {
        res.reset(new HDF5Stream);
    }


    res->current_file_name(url);

    return res;
}
//
//void close()
//{
//    SingletonHolder<std::unique_ptr<io::IOStream>>::instance()->close();
//    SingletonHolder<std::unique_ptr<io::IOStream>>::instance().reset(nullptr);
//}
//
//IOStream &global()
//{
//    return *(SingletonHolder<std::unique_ptr<io::IOStream>>::instance());
//}
//
//std::string write(std::string const &url, data_model::DataSet const &ds, size_t flag)
//{
//    return SingletonHolder<std::unique_ptr<io::IOStream>>::instance()->write(url, ds, flag);
//}
//
//data_model::DataSet load(std::string const &url)
//{
//    data_model::DataSet ds;
//    SingletonHolder<std::unique_ptr<io::IOStream>>::instance()->read(url, &ds);
//    return std::move(ds);
//}
//
//std::string cd(std::string const &url)
//{
//    return std::get<1>(SingletonHolder<std::unique_ptr<io::IOStream>>::instance()->open(url));
//}
}}// namespace simpla//namespace io

