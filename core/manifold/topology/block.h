/**
 * @file block.h
 * @author salmon
 * @date 2015-11-06.
 */

#ifndef SIMPLA_BLOCK_H
#define SIMPLA_BLOCK_H


namespace simpla
{
struct Block
{
    static constexpr int MAX_NUM_OF_VERTEX = 8;

    typedef std::int64_t id_type;

    typedef nTuple<Real, 3> coordinates_type;

    virtual size_t hash(id_type s) const = 0;

    virtual size_t max_hash(int nid = 0) const = 0;

    virtual int get_adjoin_vertrics(id_type s, id_type *res = nullptr) const = 0;

    virtual std::tuple <coordinates_type, coordinates_type> box() const = 0;


};
}

#endif //SIMPLA_BLOCK_H
