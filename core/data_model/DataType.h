/**
 * @file data_type.h
 *
 *  created on: 2014-6-2
 *      Author: salmon
 */

#ifndef DATA_TYPE_H_
#define DATA_TYPE_H_

#include <stddef.h>
#include <iostream>
#include <memory>
#include <string>
#include <tuple>
#include <type_traits>
#include <typeindex>
#include <vector>

#include "../gtl/ntuple.h"
#include "../gtl/type_traits.h"
#include "../base/Object.h"

namespace simpla { namespace traits
{
template<typename T> struct rank;
template<typename T> struct extents;
template<typename T> struct value_type;
}}//namespace simpla{namespace traits

namespace simpla { namespace data_model
{
/**
 *  @ingroup data_interface
 *
 *  \brief  Description of data type
 *
 *  @todo this class should meet the requirement of XDR
 *  http://en.wikipedia.org/wiki/External_Data_Representation#XDR_data_types
 *  see   eXternal Data Representation Standard: Protocol Specification
 *        eXternal Data Representation: Sun Technical Notes
 *        XDR: External Data Representation Standard, RFC 1014, Sun Microsystems, Inc., USC-ISI.
 *        doc/reference/xdr/
 *
 */
struct DataType : public base::SpObject
{
    SP_OBJECT_HEAD(DataType, base::SpObject);

    DataType();

    DataType(std::type_index t_index, size_t ele_size_in_byte, int ndims = 0, size_t const *dims = nullptr,
             std::string name = "");

    DataType(const DataType &other);

    DataType(DataType &&other);

    ~DataType();

    DataType &operator=(DataType const &other);

    void swap(DataType &);

    virtual std::ostream &print(std::ostream &os, int indent) const;

    bool is_valid() const;

    std::string name() const;


    size_t size() const;

    size_t size_in_byte() const;

    size_t ele_size_in_byte() const;

    int rank() const;

    DataType element_type() const;

    size_t extent(int n) const;

    void extent(size_t *d) const;

    void extent(int rank, size_t const *d);

    std::vector<size_t> const &extents() const;

    bool is_compound() const;

    bool is_array() const;

    bool is_opaque() const;

    bool is_same(std::type_index const &other) const;

    template<typename T> bool is_same() const { return is_same(std::type_index(typeid(T))); }

    void push_back(DataType &&dtype, std::string const &name, int pos = -1);

    std::vector<std::tuple<DataType, std::string, int>> const &members() const;

private:
    template<typename T> struct create_helper;
public:
    template<typename T>
    static DataType create(std::string const &s_name = "")
    {
        return create_helper<T>::create();
    }

private:
    struct pimpl_s;
    std::unique_ptr<pimpl_s> pimpl_;

};

template<typename T>
struct DataType::create_helper
{
private:
    HAS_STATIC_MEMBER_FUNCTION (datatype)

    static DataType create_(std::string const &name, std::integral_constant<bool, true>)
    {
        return T::datatype();
    }

    static DataType create_(std::string const &name, std::integral_constant<bool, false>)
    {

        typedef typename std::remove_cv<T>::type obj_type;

        typedef typename traits::value_type<obj_type>::type element_type;

        size_t ele_size_in_byte = sizeof(element_type) / sizeof(char);

        nTuple<size_t, 10> d;

        d = traits::seq_value<::simpla::traits::extents_t<obj_type> >::value;

        return std::move(

                DataType(std::type_index(typeid(element_type)),

                         ele_size_in_byte,

                         ::simpla::traits::rank<obj_type>::value,

                         &d[0],

                         name)

        );


    }

public:

    static DataType create(std::string const &name = "")
    {
        return create_(((name != "") ? name : (typeid(T).name())),
                       std::integral_constant<bool, has_static_member_function_datatype<T>::value>());
    }


};

}}//namespace simpla { namespace data_model



#endif /* DATA_TYPE_H_ */
