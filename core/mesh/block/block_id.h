//
// Created by salmon on 7/8/15.
//

#ifndef SIMPLA_BLOCK_ID_H
#define SIMPLA_BLOCK_ID_H
namespace simpla {

struct BlockID
{
    typedef std::uint64_t value_type;
private:
    value_type m_id_;
public:
    enum { LOCAL, REMOTE, GLOBAL };

    BlockID(value_type const &v) : m_id_(v)
    {
    }

    operator value_type() const
    {
        return m_id_;
    }

    value_type id() const
    { return m_id_; }


    int level() const;

    int rank() const;

    int local_id() const;

};

struct BlockIDFactory
{

};
}// namespace simpla

#endif //SIMPLA_BLOCK_ID_H
