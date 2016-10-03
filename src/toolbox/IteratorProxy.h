//
// Created by salmon on 16-5-18.
//

#ifndef SIMPLA_toolbox_ITERATORPROXY_H
#define SIMPLA_toolbox_ITERATORPROXY_H
namespace simpla { namespace toolbox
{
template<typename BaseIterator, typename Proxy>
struct IteratorProxy : public BaseIterator
{
    typedef std::result_of<Proxy(typename BaseIterator::value_type)>::type value_type;

    value_type &operator*() { return m_proxy_(BaseIterator::operator*()); }

    value_type const &operator*() const { return m_proxy_(BaseIterator::operator*()); }
};
}}
#endif //SIMPLA_toolbox_ITERATORPROXY_H
