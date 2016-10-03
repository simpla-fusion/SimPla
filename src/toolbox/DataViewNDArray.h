//
// Created by salmon on 16-5-18.
//

#ifndef SIMPLA_DATAVIEWNDARRAY_H
#define SIMPLA_DATAVIEWNDARRAY_H
namespace simpla { namespace toolbox
{
namespace detail
{
template<int NDIMS>
struct ndArrayViewHash
{
    nTuple <size_t, NDIMS> m_offsets_, m_extents_;

    typedef ndArrayViewHash<NDIMS> this_type;
public:
    ndArrayViewHash(nTuple <size_t, NDIMS> const &b, nTuple <size_t, NDIMS> const &e) : m_offsets_(b), m_extents_(e) { }

    ndArrayViewHash(this_type const &other) : m_offsets_(other.m_offsets_), m_extents_(other.m_extents_) { }

    ndArrayViewHash(this_type &&other) : m_offsets_(other.m_offsets_), m_extents_(other.m_extents_) { }

    ~ndArrayViewHash() { }


    constexpr size_t &operator()(size_t i, size_t ...others) const
    {
        static_assert(sizeof...(idx) + 1 == NDIMS);
        return get(i, &m_lower_[0], &m_upper_[0], others...);
    }

private:
    static inline constexpr size_t get(size_t res, size_t *offsets, size_t *extents, size_t i, size_t ...others)
    {
        return get(res * extents[0] + i - offsets[0], b + 1, e + 1, others...);
    }
};

template<typename T, int ND> using ndArrayView=DataView<T, size_t, detail::ndArrayViewHash<ND> >;

} //namespace detail
}}//namespace simpla { namespace toolbox

#endif //SIMPLA_DATAVIEWNDARRAY_H
