/**
 * @file IOExt.h
 * @author salmon
 * @date 2015-11-11.
 */

#ifndef SIMPLA_IO_EXT_H
#define SIMPLA_IO_EXT_H

#include <vector>
#include "IO.h"
#include "type_traits.h"
#include "DataSet.h"

namespace simpla
{


template<typename T>
std::string save(std::string const &url, T const *data, int ndims,
                 size_t const *dims, size_t flag = 0UL)
{

    data_model::DataSet res;
    res.data_type = data_model::DataType::create<T>();
    res.data_space = data_model::DataSpace::create(ndims, dims);
    res.data = std::shared_ptr<void>(const_cast<T *>(&data[0]), tags::do_nothing());

    return save(url, res, flag);
}

template<typename T>
std::string save(std::string const &url, std::vector<T> const &data, size_t flag = 0UL)
{
    size_t size = data.size();

    return save(url, &data[0], 1, &size, flag);
}
}//namespace simpla
#endif //SIMPLA_IO_EXT_H
