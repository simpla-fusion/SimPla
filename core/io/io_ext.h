/**
 * @file io_ext.h
 * @author salmon
 * @date 2015-11-11.
 */

#ifndef SIMPLA_IO_EXT_H
#define SIMPLA_IO_EXT_H

#include <vector>
#include "io.h"
#include "../gtl/type_traits.h"

namespace simpla
{
template<typename T>
std::string save(std::string const &url, std::vector<T> const &data, size_t flag = 0UL)
{
//    return save(url, make_dataset(data, ndims, dims), flag);
    DataSet res;
    res.datatype = traits::datatype<T>::create();
    typename DataSpace::index_type size = data.size();
    res.dataspace = make_dataspace(1, &size);
    res.data = std::shared_ptr<void>(const_cast<T *>(&data[0]), tags::do_nothing());


    return save(url, res, flag);
}
}//namespace simpla
#endif //SIMPLA_IO_EXT_H
