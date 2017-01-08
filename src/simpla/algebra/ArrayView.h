//
// Created by salmon on 17-1-8.
//

#ifndef SIMPLA_ARRAYVIEW_H
#define SIMPLA_ARRAYVIEW_H
namespace simpla {
namespace algebra {
namespace declare {
template <typename T, size_type NDIMS>
struct ArrayView;
}  // namespace declare {

namespace traits {
struct sub_type<declare::Array<T, 1> > {
    typedef T type;
};
struct sub_type<declare::Array<T, NDIMS> > {
    typedef declare::Array<T, NDIMS - 1> type;
};
}  // namespace traits{
namespace declare {

template <typename T, size_type NDIMS>
struct ArrayView {
    typedef ArrayView<T, NDIMS> this_type;
    typedef traits::sub_type_t<this_type> sub_type;

    index_type sub_type;

    sub_type& operator[](index_type const& s);
    sub_type const& operator[](index_type const& s) const;

    template <typename... IDX>
    decltype(auto) at(index_type const& s, IDX&&... s) {
        return operator[](s).at(std::forward<IDX>(s)...);
    }
    template <typename... IDX>
    decltype(auto) at(index_type const& s, IDX&&... s)const {
        return operator[](s).at(std::forward<IDX>(s)...);
    }
};
}  // namespace declare{

}  // namespace algebra{
}  // namespace simpla{

#endif  // SIMPLA_ARRAYVIEW_H
