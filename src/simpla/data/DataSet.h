/**
 * @file dataset.h
 *
 *  Created on: 2014-11-10
 *      Author: salmon
 */

#ifndef CORE_DATASET_DATASET_H_
#define CORE_DATASET_DATASET_H_

#include <algorithm>
#include <cstddef>
#include <list>
#include <memory>
#include <type_traits>
#include "DataSpace.h"
#include "DataType.h"
#include "simpla/utilities/type_traits.h"

namespace simpla {
namespace data {
/**
 * @addtogroup data  Dataset
 * @brief This section describes the interface of m_data set.
 *
 * @ref data_model  is a group of classes used to exchange m_data between different libraries
 * and program languages in the memory. For example, we can transfer an array
 * of particle structure in memory to hdf5 library, and write it to disk.
 */

/**
 * @ingroup data
 *
 * @brief Describe structure of m_data in the memory.
 *
 * A dataset is composed of a pointer to raw m_data , a description
 * of element m_data type (DataType), a description of memory layout of
 * m_data set (data_space),and a container of meta m_data (Properties).
 */

struct DataSet {
    std::shared_ptr<void> data;

    DataType data_type;

    DataSpace data_space;

    DataSpace memory_space;

    DataSet() : data(nullptr) {}

    DataSet(DataSet const &other)
        : data(other.data),
          data_type(other.data_type),
          data_space(other.data_space),
          memory_space(other.memory_space) {}

    DataSet(DataSet &&other)
        : data(other.data),
          data_type(other.data_type),
          data_space(other.data_space),
          memory_space(other.memory_space) {}

    virtual ~DataSet() {}

    void swap(DataSet &other) {
        std::swap(data, other.data);
        std::swap(data_type, other.data_type);
        std::swap(data_space, other.data_space);
        std::swap(memory_space, other.memory_space);
    }

    bool operator==(DataSet const &other) const { return is_equal(other.data.get()); }

    bool is_valid() const {
        return (data != nullptr) && data_type.is_valid() && data_space.is_valid() && memory_space.is_valid() &&
               (data_space.num_of_elements() == memory_space.num_of_elements());
    }

    virtual bool empty() const { return data == nullptr; }

    virtual std::ostream &print(std::ostream &os) const;

    bool is_same(void const *other) const;

    bool is_equal(void const *other) const;

    template <typename T>
    T &getValue(size_t s) {
        return reinterpret_cast<T *>(data.get())[s];
    }

    template <typename T>
    T const &getValue(size_t s) const {
        return reinterpret_cast<T *>(data.get())[s];
    }

    template <typename TV>
    TV *pointer() {
        return reinterpret_cast<TV *>(data.get());
    }

    template <typename TV>
    TV const *pointer() const {
        return reinterpret_cast<TV *>(data.get());
    }

    template <typename... Args>
    static DataSet create(Args &&... args);
};  // class dataset

//
namespace _impl {
CHECK_MEMBER_FUNCTION(has_member_function_data_set, data_set)

template <typename T>
auto create_data_set(T const &f) -> typename std::enable_if<has_member_function_data_set<T>::value, DataSet>::type {
    return f.data_set();
}

template <typename... Args>
DataSet create_data_set(DataType const &dtype, std::shared_ptr<void> const &data, Args &&... args) {
    DataSet ds;

    ds.data_type = dtype;
    ds.data = data;

    ds.memory_space = data::DataSpace::create_simple(std::forward<Args>(args)...);
    ds.data_space = ds.memory_space;

    return std::move(ds);
}

template <typename... Args>
DataSet create_data_set(DataType const &dtype) {
    DataSet ds;

    ds.data_type = dtype;
    ds.data = nullptr;
    return std::move(ds);
}

template <typename T, typename... Args>
DataSet create_data_set(T const *p, Args &&... args) {
    return create_data_set(DataType::create<T>(), std::shared_ptr<void>(reinterpret_cast<void *>(const_cast<T *>(p)),
                                                                        simpla::tags::do_nothing()),
                           std::forward<Args>(args)...);
}

template <typename T>
DataSet create_data_set(T const *p, int ndims, const size_type *d) {
    DataSet ds;

    ds.data_type = DataType::create<T>();
    ds.data = std::shared_ptr<void>(reinterpret_cast<void *>(const_cast<T *>(p)), simpla::tags::do_nothing());

    ds.memory_space = DataSpace::create_simple(ndims, d);

    ds.memory_space = ds.data_space;

    return std::move(ds);
}

template <typename T, typename... Args>
DataSet create_data_set(std::shared_ptr<T> &p, Args &&... args) {
    return create_data_set(DataType::create<T>(),
                           std::shared_ptr<void>(reinterpret_cast<void *>(p.get()), simpla::tags::do_nothing()),
                           std::forward<Args>(args)...);
}

template <typename T, typename... Args>
DataSet create_data_set(std::shared_ptr<T> const &p, Args &&... args) {
    auto ds = create_data_set(DataType::create<T>(),
                              std::shared_ptr<void>(reinterpret_cast<void *>(p.get()), simpla::tags::do_nothing()),
                              std::forward<Args>(args)...);

    return std::move(ds);
}

template <typename T>
DataSet create_data_set(std::vector<T> const &p) {
    size_t s = p.size();
    return create_data_set(&p[0], 1, &s);
}
/**@}*/
}  // namespace _impl

template <typename... Args>
DataSet DataSet::create(Args &&... args) {
    return _impl::create_data_set(std::forward<Args>(args)...);
}
}
}  // namespace simpla { namespace utilities
#endif /* CORE_DATASET_DATASET_H_ */
