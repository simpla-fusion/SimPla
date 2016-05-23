//
// Created by salmon on 16-5-18.
//

#ifndef SIMPLA_GTL_DENSEHASHMAP_H
#define SIMPLA_GTL_DENSEHASHMAP_H

namespace simpla { namespace gtl
{
template<typename Key, typename V, typename Hash>
class DenseHashMap : public std::vector<V>
{
public:
    typedef Key key_type;
    typedef V value_type;
    typedef std::vector <V> base_type;
private:
    typedef DenseHashMap<Key, V, Hash> this_type;
    Hash m_hash_;
public:

    DenseHashMap(Hash const &hash) : m_hash_(hash) { }

    template<typename ... Args>
    DenseHashMap(Hash const &hash, Args &&...args) : base_type(std::forward<Args>(args)...), m_hash_(hash) { }

    DenseHashMap(this_type const &other) : base_type(other), m_hash_(other.hash) { }

    DenseHashMap(this_type &&other) : base_type(other), m_hash_(other.hash) { }

    ~DenseHashMap() { }

    this_type &operator=(this_type const &other)
    {
        this_type(other).swap(*this);
        return *this;
    }

    void swap(this_type &other)
    {
        std::swap(m_hash_, other.m_hash_);
        base_type::swap(other);
    }


    value_type &operator[](key_type const &k) { return base_type::operator[](m_hash_(k)); }

    value_type &operator[](key_type const &k) const { return base_type::operator[](m_hash_(k)); }

};
}}
#endif //SIMPLA_GTL_DENSEHASHMAP_H
