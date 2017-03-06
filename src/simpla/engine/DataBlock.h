//
// Created by salmon on 16-11-2.
//

#ifndef SIMPLA_DATABLOCK_H
#define SIMPLA_DATABLOCK_H

#include <simpla/SIMPLA_config.h>
#include <simpla/concept/Printable.h>
#include <simpla/concept/Serializable.h>
#include <simpla/toolbox/FancyStream.h>

namespace simpla {
namespace engine {
/**
 *  Base class of Data Blocks (pure virtual)
 */
class MeshBlock;

class DataBlock {
    SP_OBJECT_BASE(DataBlock);

   public:
    DataBlock() {}
    virtual ~DataBlock() {}
    bool empty() const { return raw_data() == nullptr; }

    bool isInitialized() const { return true; }
    virtual std::type_info const &value_type_info() const { return typeid(Real); };
    virtual int entity_type() const { return 0; };
    virtual int dof() const { return 1; };
    virtual void *raw_data() { return nullptr; };
    virtual void const *raw_data() const { return nullptr; };
    virtual void Clear(){};
};
template <typename U, int IFORM, int DOF>
class DefaultDataBlock : public DataBlock {
    typedef U value_type;

   public:
    DefaultDataBlock(std::shared_ptr<value_type> d = nullptr, size_type s = 0) : m_data_(d), m_size_(s) {}
    virtual ~DefaultDataBlock() {}
    virtual std::type_info const &value_type_info() const { return typeid(value_type); };
    virtual int entity_type() const { return IFORM; };
    virtual int dof() const { return DOF; };
    virtual void *raw_data() { return m_data_.get(); };
    virtual void const *raw_data() const { return m_data_.get(); };

    virtual void Clear() {
        if (m_data_ != nullptr && m_size_ > 0) { memset(m_data_.get(), 0, sizeof(value_type) * m_size_); }
    };

    std::shared_ptr<value_type> data() { return m_data_; };
    size_type size() const { return m_size_; }

   private:
    std::shared_ptr<value_type> m_data_;
    size_type m_size_;
};
template <typename...>
class DataBlockAdapter;

/**
   * concept::Serializable
   *    virtual void load(data::DataTable const &) =0;
   *    virtual void save(data::DataTable *) const =0;
   *
   * concept::Printable
   *    virtual std::ostream &print(std::ostream &os, int indent) const =0;
   *
   * Object
   *    virtual bool is_deployed() const =0;
   *    virtual bool is_valid() const =0;
   */
template <typename U>
class DataBlockAdapter<U> : public DataBlock, public U {
    SP_OBJECT_HEAD(DataBlockAdapter<U>, DataBlock);
    typedef algebra::traits::value_type_t<U> value_type;

   public:
    template <typename... Args>
    explicit DataBlockAdapter(Args &&... args) : U(std::forward<Args>(args)...) {}
    ~DataBlockAdapter() {}
    virtual std::type_info const &value_type_info() const { return typeid(algebra::traits::value_type_t<U>); };
    virtual int entity_type() const { return algebra::traits::iform<U>::value; }
    virtual int dof() const { return algebra::traits::dof<U>::value; }
    virtual void Load(data::DataTable const &d){/* Load(*this, d); */};
    virtual void Save(data::DataTable *d) const {/* Save(*this, d); */};
    virtual std::ostream &Print(std::ostream &os, int indent) const {
        os << " type = \'" << value_type_info().name() << "\' "
           << ", entity type = " << (entity_type()) << ", GetDOF = " << (dof()) << ", GetDataBlock = {";
        U::Print(os, indent + 1);
        os << "}";
        return os;
    }
    virtual void *raw_data() { return reinterpret_cast<void *>(U::data()); };
    virtual void const *raw_data() const { return reinterpret_cast<void const *>(U::data()); };

    template <typename... Args>
    static std::shared_ptr<DataBlock> Create(Args &&... args) {
        return std::dynamic_pointer_cast<DataBlock>(std::make_shared<DataBlockAdapter<U>>(std::forward<Args>(args)...));
    }
    virtual void Clear() { U::Clear(); }
};

// template<typename V, int IFORM = VERTEX, int DOF = 1, bool SLOW_FIRST = false>
// using DataBlockArray=
// DataBlockAdapter<
//        Array < V,
//        SIMPLA_MAXIMUM_DIMENSION + (((IFORM == VERTEX || IFORM == VOLUME) && DOF == 1) ? 0 : 1), SLOW_FIRST>>;
//
// template<typename TV, int IFORM, int DOF = 1>
// class DataBlockArray : public DataBlock, public data::DataEntityNDArray<TV>
//{
// public:
//    typedef DataBlockArray<TV, IFORM, DOF> block_array_type;
//    typedef data::DataEntityNDArray<TV> data_entity_traits;
//    typedef TV value_type;
//
// SP_OBJECT_HEAD(block_array_type, DataBlock);
//
//    template<typename ...Args>
//    explicit DataBlockArray(Args &&...args) : DataBlock(), data_entity_traits(std::forward<Args>(args)...) {}
//
//    virtual ~DataBlockArray() {}
//
//    virtual bool isValid() { return data_entity_traits::isValid(); };
//
//    virtual std::type_info const &GetValueTypeInfo() const { return typeid(value_type); };
//
//    virtual int GetIFORM() const { return IFORM; }
//
//    virtual int GetDOF() const { return DOF; }
//
//    virtual void Load(data::DataTable const &) { UNIMPLEMENTED; };
//
//    virtual void Save(data::DataTable *) const { UNIMPLEMENTED; };
//
//    virtual std::ostream &Print(std::ostream &os, int indent) const
//    {
//        os << " type = \'" << GetValueTypeInfo().GetName() << "\' "
//           << ", entity type = " << static_cast<int>(GetIFORM())
//           << ", GetDataBlock = {";
//        data_entity_traits::Print(os, indent + 1);
//        os << "}";
//        return os;
//    }
//
//    virtual std::shared_ptr<DataBlock> clone(std::shared_ptr<RectMesh> const &m, void *p = nullptr)
//    {
//        return create(m, static_cast<value_type *>(p));
//    };
//
//
//    static std::shared_ptr<DataBlock>
//    create(std::shared_ptr<RectMesh> const &m, value_type *p = nullptr)
//    {
//        index_type n_dof = DOF;
//        int ndims = 3;
//        if (IFORM == EDGE || IFORM == FACE)
//        {
//            n_dof *= 3;
//            ++ndims;
//        }
//        auto b = m->outer_index_box();
//        index_type lo[4] = {std::get<0>(b)[0], std::get<0>(b)[1], std::Get<0>(b)[2], 0};
//        index_type hi[4] = {std::get<1>(b)[0], std::Get<1>(b)[1], std::get<0>(b)[2], n_dof};
//        return std::dynamic_pointer_cast<DataBlock>(std::make_shared<this_type>(p, ndims, lo, hi));
//    };
//
//    virtual void Intialize()
//    {
//        base_type::Intialize();
//        data_entity_traits::Intialize();
//    };
//
//    virtual void PreProcess() { data_entity_traits::update(); };
//
//    virtual void update() { data_entity_traits::update(); };
//
//    virtual void Finalizie()
//    {
//        data_entity_traits::Finalizie();
//        base_type::Finalizie();
//    };
//
//    virtual void Clear() { data_entity_traits::Clear(); }
//
//    virtual void Sync(std::shared_ptr<DataBlock>, bool only_ghost = true) { UNIMPLEMENTED; };
//
//
//    template<typename ...Args>
//    value_type &get(Args &&...args) { return data_entity_traits::Get(std::forward<Args>(args)...); }
//
//    template<typename ...Args>
//    value_type const &Get(Args &&...args) const { return data_entity_traits::get(std::forward<Args>(args)...); }
//
//
//    EntityIdRange Range() const
//    {
//        EntityIdRange res;
//        index_tuple lower, upper;
//        lower = data_entity_traits::index_lower();
//        upper = data_entity_traits::index_upper();
//        res.append(MeshEntityIdCoder::make_range(lower, upper, GetIFORM()));
//        return res;
//    }
//
// private:
//    index_tuple m_ghost_width_{{0, 0, 0}};
//};
}
}  // namespace simpla { namespace mesh

#endif  // SIMPLA_DATABLOCK_H
