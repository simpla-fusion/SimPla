/**
 * @file DataType.h
 *
 *  created on: 2014-6-2
 *      Author: salmon
 */

#ifndef DATA_TYPE_H_
#define DATA_TYPE_H_

#include <simpla/SIMPLA_config.h>

#include <stddef.h>
#include <iostream>
#include <memory>
#include <string>
#include <tuple>
#include <type_traits>
#include <typeindex>
#include <vector>

#include <simpla/algebra/Algebra.h>
#include <simpla/algebra/nTuple.h>
#include <simpla/utilities/type_traits.h>

namespace simpla {
namespace data {
/**
 *  @ingroup data
 *
 *  \brief  Description of m_data type
 *
 *  @todo this class should meet the requirement of XDR
 *  http://en.wikipedia.org/wiki/External_Data_Representation#XDR_data_types
 *  see   eXternal Data Representation Standard: Protocol Specification
 *        eXternal Data Representation: Sun Technical Notes
 *        XDR: External Data Representation Standard, RFC 1014, Sun Microsystems, Inc., USC-ISI.
 *        doc/reference/xdr/
 *
 */
struct DataType {
    DataType();

    DataType(std::type_index t_index, int ele_size_in_byte, int ndims = 0,
             size_type const *dims = nullptr, std::string name = "");

    DataType(const DataType &other);

    DataType(DataType &&other);

    ~DataType();

    DataType &operator=(DataType const &other);

    void swap(DataType &);

    virtual std::ostream &print(std::ostream &os, int indent) const;

    bool is_valid() const;

    virtual std::string name() const;

    int number_of_entities() const;

    void size_in_byte(int);

    int size_in_byte() const;

    int ele_size_in_byte() const;

    int rank() const;

    DataType element_type() const;

    int extent(int n) const;

    void extent(int *d) const;

    void extent(int rank, const int *d);

    std::vector<size_t> const &extents() const;

    bool is_compound() const;

    bool is_array() const;

    bool is_opaque() const;

    bool is_same(std::type_index const &other) const;

    template <typename T>
    bool is_same() const {
        return is_same(std::type_index(typeid(T)));
    }

    int push_back(DataType const &dtype, std::string const &name, index_type offset = -1);

    std::vector<std::tuple<DataType, std::string, int>> const &members() const;

   private:
    template <typename T>
    struct create_helper;

   public:
    template <typename T>
    static DataType create(std::string const &s_name = "") {
        return create_helper<T>::create();
    }

   private:
    struct pimpl_s;
    std::unique_ptr<pimpl_s> pimpl_;
};

CHECK_STATIC_FUNCTION_MEMBER(has_static_member_function_data_type, data_type)

CHECK_STATIC_FUNCTION_MEMBER(has_static_member_function_class_name, class_name);

namespace traits {

template <typename T, class Enable = void>
struct type_id {
   private:
    static std::string name_(std::true_type) { return T::class_name(); }

    static std::string name_(std::false_type) { return "unknown"; }

   public:
    static std::string name() {
        return name_(
            std::integral_constant<bool, has_static_member_function_class_name<T>::value>());
    }
};

template <int I>
struct type_id<std::integral_constant<size_t, I>> {
    static std::string name() { return "[" + simpla::type_cast<std::string>(I) + "]"; }
};

template <int I>
struct type_id<std::integral_constant<int, I>> {
    static std::string name() {
        return std::string("[") + simpla::traits::type_cast<int, std::string>::eval(I) + "]";
    }
};

CHECK_BOOLEAN_TYPE_MEMBER(is_self_describing, is_self_describing)
//
// namespace detail {
//
// template <typename T>
// struct check_static_bool_member_is_self_describing {
//   private:
//    typedef std::true_type yes;
//    typedef std::false_type no;
//
//    template <typename U>
//    static auto test(int, std::enable_if_t<is_self_describing<U>::value> * = nullptr)
//        -> std::integral_constant<bool, U::is_self_describing>;
//
//    template <typename>
//    static no test(...);
//
//   public:
//    static constexpr bool value = !std::is_same<decltype(test<T>(0)), no>::value;
//};
//}
template <typename T>
struct type_id<T, typename std::enable_if_t<is_self_describing<T>::value>> {
    static std::string name() { return T::name()(); }

    static auto data_type() -> decltype(T::data_type()) { return T::data_type(); }
};

template <typename T, int N>
struct type_id<T[N], void> {
    static std::string name() {
        return type_id<T[N]>::name() + "[" + simpla::traits::type_cast<int, std::string>::eval(N) +
               "]";
    }

    static auto data_type() -> decltype(T::data_type()) { return T::data_type(); }
};

template <typename T, typename... Others>
struct type_id_list {
    static std::string name() {
        return type_id_list<T>::name() + "," + type_id_list<Others...>::name();
    }
};

template <typename T>
struct type_id_list<T> {
    static std::string name() { return type_id<T>::name(); }
};

#define DEFINE_TYPE_ID_NAME(_NAME_)                   \
    template <>                                       \
    struct type_id<_NAME_> {                          \
        static std::string name() { return #_NAME_; } \
    };

DEFINE_TYPE_ID_NAME(double)

DEFINE_TYPE_ID_NAME(float)

DEFINE_TYPE_ID_NAME(int)

DEFINE_TYPE_ID_NAME(long)

#undef DEFINE_TYPE_ID_NAME
}
template <typename T>
struct DataType::create_helper {
   private:
    static DataType create_(std::string const &name, std::integral_constant<bool, true>) {
        return traits::type_id<T>::data_type();
    }

    static DataType create_(std::string const &name, std::integral_constant<bool, false>) {
        typedef typename std::remove_cv<T>::type obj_type;

        typedef typename algebra::traits::value_type<obj_type>::type element_type;

        int ele_size_in_byte = sizeof(obj_type) / sizeof(char);

        nTuple<size_type, 10> d;

        //        d = traits::seq_value<algebra::traits::extents<obj_type> >::value;

        return std::move(DataType(std::type_index(typeid(obj_type)), ele_size_in_byte,
                                  algebra::traits::rank<obj_type>::value, &d[0], name)

                             );
    }

   public:
    static DataType create(std::string const &name = "") {
        return create_(
            ((name != "") ? name : (typeid(T).name())),
            std::integral_constant<
                bool, has_static_member_function_data_type<traits::type_id<T>>::value>());
    }
};

template <typename T, int N>
struct DataType::create_helper<T[N]> {
    static DataType create(std::string const &name = "") {
        typedef typename std::remove_cv<T>::type obj_type;

        typedef typename algebra::traits::value_type<obj_type>::type element_type;

        int ele_size_in_byte = sizeof(element_type) / sizeof(char);

        size_type d = N;

        return std::move(
            DataType(std::type_index(typeid(element_type)), ele_size_in_byte, 1, &d, name));
    }
};

template <typename T, int N, int M>
struct DataType::create_helper<T[N][M]> {
    static DataType create(std::string const &name = "") {
        typedef typename std::remove_cv<T>::type obj_type;

        typedef typename algebra::traits::value_type<obj_type>::type element_type;

        int ele_size_in_byte = sizeof(element_type) / sizeof(char);

        size_type d[] = {N, M};

        return std::move(
            DataType(std::type_index(typeid(element_type)), ele_size_in_byte, 2, d, name)

                );
    }
};
}  // namespace data_model
}  // namespace simpla

#endif /* DATA_TYPE_H_ */
