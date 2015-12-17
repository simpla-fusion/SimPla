/**
 * @file dataset.h
 *
 *  Created on: 2014-11-10
 *      Author: salmon
 */

#ifndef CORE_DATASET_DATASET_H_
#define CORE_DATASET_DATASET_H_

#include <list>
#include <stddef.h>
#include <algorithm>
#include <memory>
#include <type_traits>

#include "../gtl/check_concept.h"
#include "../gtl/type_traits.h"
#include "../base/object.h"
#include "dataspace.h"
#include "datatype.h"

namespace simpla
{
/**
 * @addtogroup data_model Dataset
 * @brief This section describes the interface of data set.
 *
 * @ref data_model  is a group of classes used to exchange data between different libraries
 * and program languages in the memory. For example, we can transfer an array
 * of particle structure in memory to hdf5 library, and write it to disk.
 */

/**
 * @ingroup data_model
 *
 * @brief Describe structure of data in the memory.
 *
 * A DataSet is composed of a pointer to raw data , a description
 * of element data type (DataType), a description of memory layout of
 * data set (DataSpace),and a container of meta data (Properties).
 */

struct DataSet
{
    std::shared_ptr<void> data;

    DataType datatype;

    DataSpace dataspace;

    DataSpace memory_space;


    DataSet() : data(nullptr) { }

    DataSet(DataSet const &other) :
            data(other.data),
            datatype(other.datatype),
            dataspace(other.dataspace),
            memory_space(other.memory_space)
    {
    }


    virtual ~DataSet() { }

    void swap(DataSet &other)
    {
        std::swap(data, other.data);
        std::swap(datatype, other.datatype);
        std::swap(dataspace, other.dataspace);
        std::swap(memory_space, other.memory_space);
    }

    bool operator==(DataSet const &other) const { return is_equal(other.data.get()); }

    bool is_valid() const
    {
        return (data != nullptr)
               && datatype.is_valid()
               && dataspace.is_valid()
               && memory_space.is_valid()
               && (dataspace.num_of_elements() == memory_space.num_of_elements());
    }

    virtual bool empty() const { return data == nullptr; }

    virtual std::ostream &print(std::ostream &os) const;

    bool is_same(void const *other) const;

    bool is_equal(void const *other) const;

    template<typename T> T &get_value(size_t s) { return reinterpret_cast<T *>( data.get())[s]; }

    template<typename T> T const &get_value(size_t s) const { return reinterpret_cast<T *>( data.get())[s]; }


    template<typename TV> TV *pointer() { return reinterpret_cast<TV *>(data.get()); }

    template<typename TV> TV const *pointer() const { return reinterpret_cast<TV *>(data.get()); }


}; //class DataSet





namespace traits
{


//
namespace _impl
{
HAS_MEMBER_FUNCTION(dataset)
}  // namespace _impl
//
template<typename T>
auto make_dataset(T &d) ->
typename std::enable_if<_impl::has_member_function_dataset<T>::value,
        decltype(d.dataset())>::type
{
    return d.dataset();
}

template<typename T>
auto make_dataset(T const &d) ->
typename std::enable_if<_impl::has_member_function_dataset<T>::value,
        decltype(d.dataset())>::type
{
    return d.dataset();
}


DataSet make_dataset(DataType const &dtype, std::shared_ptr<void> const &p, size_t rank,
                     size_t const *dims = nullptr);

DataSet make_dataset(DataType const &dtype);

template<typename T>
DataSet make_dataset()
{
    return make_dataset(traits::datatype<T>::create());
}

template<typename T>
DataSet make_dataset(T const *p, size_t rank, size_t const *dims = nullptr)
{
    return make_dataset(traits::datatype<T>::create(),
                        std::shared_ptr<void>(reinterpret_cast<void *>(const_cast<T *>(p)), tags::do_nothing()),
                        rank, dims);
}

template<typename T>
DataSet make_dataset(std::shared_ptr<T> &p, size_t rank, size_t const *dims = nullptr)
{
    return make_dataset(traits::datatype<T>::create(),
                        std::shared_ptr<void>(reinterpret_cast<void *>(p.get()), tags::do_nothing()),
                        rank, dims);
}


template<typename T>
DataSet make_dataset(std::vector<T> const &p)
{
    return make_dataset(&p[0], p.size());
}
/**@}*/
} // namespace traits
}  // namespace simpla

#endif /* CORE_DATASET_DATASET_H_ */
