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
#include "../gtl/properties.h"
#include "../gtl/type_traits.h"
#include "dataspace.h"
#include "datatype.h"

namespace simpla
{
/**
 * @addtogroup dataset Dataset
 * @brief This section describes the interface of data set.
 *
 * @ref dataset  is a group of classes used to exchange data between different libraries
 * and program languages in the memory. For example, we can transfer an array
 * of particle structure in memory to hdf5 library, and save it to disk.
 */

/**
 * @ingroup dataset
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

    Properties properties;

    DataSet() : data(nullptr) { }

    DataSet(DataSet const &other) :
            data(other.data),
            datatype(other.datatype),
            dataspace(other.dataspace),
            memory_space(other.memory_space),
            properties(other.properties) { }

    DataSet(DataSet &&other) :
            data(other.data),
            datatype(other.datatype),
            dataspace(other.dataspace),
            memory_space(other.memory_space),
            properties(other.properties) { }

    virtual ~DataSet() { }

    void swap(DataSet &other)
    {
        std::swap(data, other.data);
        std::swap(datatype, other.datatype);
        std::swap(dataspace, other.dataspace);
        std::swap(memory_space, other.memory_space);
        std::swap(properties, other.properties);
    }

    bool operator==(DataSet const &other) const { return is_equal(other.data.get()); }

    virtual bool is_valid() const
    {
        return (data != nullptr) && (datatype.is_valid()) && (dataspace.size() == memory_space.size());
    }

    virtual bool empty() const { return data == nullptr; }

    virtual void deploy();

    virtual std::ostream &print(std::ostream &os) const;

    virtual void clear();

    virtual void copy(void const *other);

    virtual bool is_same(void const *other) const;

    virtual bool is_equal(void const *other) const;

    template<typename T> T &get_value(size_t s) { return reinterpret_cast<T *>( data.get())[s]; }

    template<typename T> T const &get_value(size_t s) const { return reinterpret_cast<T *>( data.get())[s]; }


    std::map<size_t, std::tuple<DataSpace, DataSet>> children;
}; //class DataSet
//template<typename T, typename ...Args>
//DataSet make_dataset(Args &&...args)
//{
//    DataSet res;
//    res.datatype = traits::datatype<T>::create();
//    res.dataspace = make_dataspace(std::forward<Args>(args)...);
//    res.deploy();
//    return std::move(res);
//};
namespace traits
{
inline void deploy(DataSet *d) { d->deploy(); }

template<typename T> inline T &get_value(DataSet &d, size_t s) { return d.template get_value<T>(s); }

template<typename T> inline T const &get_value(DataSet const &d, size_t s) { return d.template get_value<T>(s); }

}//namespace traits
//
//namespace _impl
//{
//HAS_MEMBER_FUNCTION(dataset)
//}  // namespace _impl
//
//template<typename T>
//auto make_dataset(T &d) ->
//typename std::enable_if<_impl::has_member_function_dataset<T>::value,
//		decltype(d.dataset())>::type
//{
//	return std::move(d.dataset());
//}
//
//template<typename T>
//auto make_dataset(T *d) ->
//typename std::enable_if<_impl::has_member_function_dataset<T>::value,
//		decltype(d->dataset())>::type
//{
//	return std::move(d->dataset());
//}
//
//template<typename T, typename TI>
//DataSet make_dataset(T *p, int rank, TI const *dims, Properties const &prop =
//Properties())
//{
//
//	DataSet res;
//
//	res.datatype = traits::datatype<T>::create();
//	res.dataspace = DataSpace::create_simple(rank, dims);
////	res.data = std::shared_ptr<void>(
////			const_cast<void*>(reinterpret_cast<typename std::conditional<
////					std::is_const<T>::value, void const *, void *>::type>(p)),
////			do_nothing());
//	res.properties = prop;
//
//	return std::move(res);
//}
//
//template<typename T, typename TI>
//DataSet make_dataset(std::shared_ptr<T> p, int rank, TI const *dims,
//		Properties const &prop = Properties())
//{
//	DataSet res;
//	res.data = p;
//	res.datatype = traits::datatype<T>::create();
//	res.dataspace = make_dataspace(rank, dims);
//	res.properties = prop;
//
//	return std::move(res);
//}
//
//template<typename T>
//DataSet make_dataset(std::vector<T> const &p)
//{
//
//	DataSet res;
//	long num = p.size();
//	res.datatype = traits::datatype<T>::create();
//	res.dataspace = DataSpace::create_simple(1, &num);
//	res.data = std::shared_ptr<void>(
//			const_cast<void*>(reinterpret_cast<void const *>(&p[0])),
//			do_nothing());
//
//	return std::move(res);
//}
/**@}*/

}  // namespace simpla

#endif /* CORE_DATASET_DATASET_H_ */
