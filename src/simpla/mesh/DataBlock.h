//
// Created by salmon on 16-11-2.
//

#ifndef SIMPLA_DATABLOCK_H
#define SIMPLA_DATABLOCK_H

#include <simpla/SIMPLA_config.h>
#include <simpla/toolbox/PrettyStream.h>
#include <simpla/concept/Serializable.h>
#include <simpla/concept/Printable.h>
#include <simpla/data/DataEntityNDArray.h>
#include "MeshCommon.h"
#include "EntityId.h"
#include "MeshBlock.h"

namespace simpla { namespace mesh
{
/**
 *  Base class of Data Blocks (pure virtual)
 */
class MeshBlock;

class DataBlock :
        public concept::Serializable,
        public concept::Printable,
        public concept::LifeControllable
{
public:
    SP_OBJECT_BASE(DataBlock);

    DataBlock() {}

    virtual ~DataBlock() {}

    virtual bool is_valid()=0;

    virtual std::type_info const &value_type_info() const =0;

    virtual size_type entity_type() const =0;

    virtual void load(data::DataTable const &) =0;

    virtual void save(data::DataTable *) const =0;

    virtual std::ostream &print(std::ostream &os, int indent) const =0;

    virtual void clear()=0;

    virtual void update()=0;

    virtual void pre_process()=0;

    virtual std::shared_ptr<DataBlock> clone(std::shared_ptr<MeshBlock> const &m, void *p = nullptr)=0;

};

template<typename TV, size_type IFORM, size_type DOF = 1>
class DataBlockArray : public DataBlock, public data::DataEntityNDArray<TV>
{
public:
    typedef DataBlockArray<TV, IFORM, DOF> block_array_type;
    typedef data::DataEntityNDArray<TV> data_entity_type;
    typedef TV value_type;

SP_OBJECT_HEAD(block_array_type, DataBlock);

    template<typename ...Args>
    explicit DataBlockArray(Args &&...args) : DataBlock(), data_entity_type(std::forward<Args>(args)...) {}

    virtual ~DataBlockArray() {}

    virtual bool is_valid() { return data_entity_type::is_valid(); };

    virtual std::type_info const &value_type_info() const { return typeid(value_type); };

    virtual size_type entity_type() const { return IFORM; }

    virtual size_type dof() const { return DOF; }

    virtual void load(data::DataTable const &) { UNIMPLEMENTED; };

    virtual void save(data::DataTable *) const { UNIMPLEMENTED; };

    virtual std::ostream &print(std::ostream &os, int indent) const
    {
        os << " type = \'" << value_type_info().name() << "\' "
           << ", entity type = " << static_cast<int>(entity_type())
           << ", data_block = {";
        data_entity_type::print(os, indent + 1);
        os << "}";
        return os;
    }

    virtual std::shared_ptr<DataBlock> clone(std::shared_ptr<MeshBlock> const &m, void *p = nullptr)
    {
        return create(m, static_cast<value_type *>(p));
    };


    static std::shared_ptr<DataBlock>
    create(std::shared_ptr<MeshBlock> const &m, value_type *p = nullptr)
    {
        index_type n_dof = DOF;
        int ndims = 3;
        if (IFORM == EDGE || IFORM == FACE)
        {
            n_dof *= 3;
            ++ndims;
        }
        auto b = m->outer_index_box();
        index_type lo[4] = {std::get<0>(b)[0], std::get<0>(b)[1], std::get<0>(b)[2], 0};
        index_type hi[4] = {std::get<1>(b)[0], std::get<1>(b)[1], std::get<0>(b)[2], n_dof};
        return std::dynamic_pointer_cast<DataBlock>(std::make_shared<this_type>(p, ndims, lo, hi));
    };

    virtual void deploy()
    {
        base_type::deploy();
        data_entity_type::deploy();
    };

    virtual void pre_process() { data_entity_type::update(); };

    virtual void update() { data_entity_type::update(); };

    virtual void destroy()
    {
        data_entity_type::destroy();
        base_type::destroy();
    };

    virtual void clear() { data_entity_type::clear(); }

    virtual void sync(std::shared_ptr<DataBlock>, bool only_ghost = true) { UNIMPLEMENTED; };


    template<typename ...Args>
    value_type &get(Args &&...args) { return data_entity_type::get(std::forward<Args>(args)...); }

    template<typename ...Args>
    value_type const &get(Args &&...args) const { return data_entity_type::get(std::forward<Args>(args)...); }


    EntityIdRange range() const
    {
        EntityIdRange res;
        index_tuple lower, upper;
        lower = data_entity_type::index_lower();
        upper = data_entity_type::index_upper();
        res.append(MeshEntityIdCoder::make_range(lower, upper, entity_type()));
        return res;
    }

private:
    index_tuple m_ghost_width_{{0, 0, 0}};
};
}}//namespace simpla { namespace mesh

#endif //SIMPLA_DATABLOCK_H

