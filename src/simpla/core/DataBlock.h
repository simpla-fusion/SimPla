//
// Created by salmon on 16-11-2.
//

#ifndef SIMPLA_DATABLOCK_H
#define SIMPLA_DATABLOCK_H
#include "simpla/SIMPLA_config.h"

#include "DataEntity.h"
#include "simpla/algebra/Array.h"
namespace simpla {
namespace data {
/**
 *  Base class of Data Blocks (pure virtual)
 */

template <typename V>
struct DataBlock : public DataEntity, public Array<V> {
    SP_DEFINE_FANCY_TYPE_NAME(DataBlock<V>, DataEntity);
    typedef V value_type;
    typedef Array<V> array_type;

   protected:
    explicit DataBlock() = default;

    template <typename... Args>
    explicit DataBlock(Args &&... args) : Array<V>(std::forward<Args>(args)...){};

   public:
    ~DataBlock() override = default;

    template <typename... Args>
    static std::shared_ptr<this_type> New(Args &&... args);

    std::type_info const &value_type_info() const override { return typeid(value_type); };
    size_type value_alignof() const override { return alignof(value_type); };

    size_type size() const override { return array_type::size(); };

    int GetNDIMS() const override { return array_type::GetNDIMS(); };
    int GetIndexBox(index_type *lo, index_type *hi) const override { return array_type::GetIndexBox(lo, hi); };

    void *GetPointer() override { return Array<V>::get(); }
    void const *GetPointer() const override { return Array<V>::get(); }

    void Clear() override { array_type::Clear(); };

    //    size_type CopyIn(DataBlock const &other, index_type const *lo, index_type const *hi) override {
    //        size_type count = 0;
    //        if (auto *p = dynamic_cast<ArrayBase const *>(&other)) { count = array_type::CopyIn(*p, lo, hi); }
    //        return count;
    //    };
    //    size_type CopyOut(DataBlock &other, index_type const *lo, index_type const *hi) const override {
    //        size_type count = 0;
    //        if (auto *p = dynamic_cast<ArrayBase *>(&other)) { count = array_type::CopyOut(*p, lo, hi); }
    //        return count;
    //    }
    std::ostream &Print(std::ostream &os, int indent) const override {
        os << "Array<" << simpla::traits::type_name<value_type>::value() << ">" << Array<value_type>::GetIndexBox();
        return os;
    }
};
template <typename U>
std::shared_ptr<data::DataBlock<U>> make_data_block(U *p, int rank, index_type const *lo, index_type const *hi) {
    return DataBlock<U>::New(p, rank, lo, hi);
}

template <typename U>
std::shared_ptr<data::DataBlock<U>> make_data_block(std::shared_ptr<U> const &p, int rank, index_type const *lo,
                                                    index_type const *hi) {
    return DataBlock<U>::New(p, rank, lo, hi);
}
template <typename U>
std::shared_ptr<data::DataBlock<U>> make_data_block(std::shared_ptr<U> const &p, index_box_type const &ibox) {
    return DataBlock<U>::New(p, 3, &std::get<0>(ibox)[0], &std::get<1>(ibox)[0]);
}
template <typename V>
template <typename... Args>
std::shared_ptr<DataBlock<V>> DataBlock<V>::New(Args &&... args) {
    return std::shared_ptr<DataBlock<V>>(new DataBlock<V>(std::forward<Args>(args)...));
};
//
// template <typename... Others>
// class DataMultiArray;
//
// template <typename U, typename... Others>
// class DataMultiArray<simpla::Array<U, Others...>> : public DataBlock {
//   public:
//    typedef Array<U, Others...> array_type;
//    typedef DataMultiArray<array_type> multi_array_type;
//    SP_DEFINE_FANCY_TYPE_NAME(multi_array_type, DataBlock);
//
//    typedef U value_type;
//
//   protected:
//    explicit DataMultiArray(unsigned long depth) : m_data_(depth) {}
//
//   public:
//    ~DataMultiArray() override = default;
//
//    static std::shared_ptr<DataMultiArray> Create(unsigned long depth) {
//        return std::shared_ptr<DataMultiArray>(new DataMultiArray(depth));
//    }
//
//    size_type size() const { return m_data_.size(); }
//    std::type_info const &value_type_info() const override { return typeid(value_type); };
//    size_type GetDepth() const override { return m_data_.size(); }
//
//    void SetArray(int depth, array_type d) { array_type(d).swap(m_data_.at(depth)); }
//
//    array_type *GetEntity(int depth = 0) { return &m_data_.at(depth); }
//    array_type const *GetEntity(int depth = 0) const { return &m_data_.at(depth); }
//    array_type &GetArray(int depth = 0) { return m_data_.at(depth); }
//    array_type const &GetArray(int depth = 0) const { return m_data_.at(depth); }
//
//    auto &at(int depth = 0) { return m_data_.at(depth); }
//    auto const &at(int depth = 0) const { return m_data_.at(depth); }
//    array_type &operator[](int depth) { return m_data_.at(depth); }
//    array_type const &operator[](int depth) const { return m_data_.at(depth); }
//    void DeepCopy(std::shared_ptr<this_type> const &other) {
//        if (other == nullptr) { return; }
//        if (m_data_.size() < other->size()) { m_data_.resize(other->m_data_.size()); }
//        for (int i = 0; i < m_data_.size(); ++i) { m_data_.at(i).DeepCopy(other->m_data_.at(i).get()); }
//    }
//    void Clear() override {
//        for (int i = 0; i < m_data_.size(); ++i) { m_data_.at(i).Clear(); };
//    };
//
//   private:
//    std::vector<array_type> m_data_;
//};

// template <typename...>
// class DataBlockAdapter;
//
///**
//   * concept::Serializable
//   *    virtual void load(data::DataTable const &) =0;
//   *    virtual void save(data::DataTable *) const =0;
//   *
//   * concept::Printable
//   *    virtual std::ostream &print(std::ostream &os, int indent) const =0;
//   *
//   * Object
//   *    virtual bool is_deployed() const =0;
//   *    virtual bool is_valid() const =0;
//   */
// template <typename U>
// class DataBlockAdapter<U> : public DataBlock, public U {
//    SP_DEFINE_FANCY_TYPE_NAME(DataBlockAdapter<U>, DataBlock);
//    typedef algebra::traits::value_type_t<U> value_type;
//
//   public:
//    template <typename... Args>
//    explicit DataBlockAdapter(Args &&... args) : U(std::forward<Args>(args)...) {}
//    ~DataBlockAdapter() {}
//    virtual std::type_info const &value_type_info() const { return typeid(algebra::traits::value_type_t<U>); };
//    virtual int entity_type() const { return algebra::traits::iform<U>::value; }
//    virtual int dof() const { return algebra::traits::dof<U>::value; }
//    virtual std::ostream &Print(std::ostream &os, int indent) const {
//        os << " value_type_info = \'" << value_type_info().name() << "\' "
//           << ", entity value_type_info = " << (entity_type()) << ", GetDOF = " << (dof()) << ", GetDataBlock = {";
//        U::Print(os, indent + 1);
//        os << "}";
//        return os;
//    }
//    virtual void *raw_data() { return reinterpret_cast<void *>(U::data()); };
//    virtual void const *raw_data() const { return reinterpret_cast<void const *>(U::data()); };
//
//    template <typename... Args>
//    static std::shared_ptr<DataBlock> Create(Args &&... args) {
//        return
//        std::dynamic_pointer_cast<DataBlock>(std::make_shared<DataBlockAdapter<U>>(std::forward<Args>(args)...));
//    }
//    virtual void Clear() { U::Clear(); }
//};

// template<typename V, int IFORM = NODE, int DOF = 1, bool SLOW_FIRST = false>
// using DataBlockArray=
// DataBlockAdapter<
//        Array < V,
//        SIMPLA_MAXIMUM_DIMENSION + (((IFORM == NODE || IFORM == CELL) && DOF == 1) ? 0 : 1), SLOW_FIRST>>;
//
// template<typename TV, int IFORM, int DOF = 1>
// class DataBlockArray : public DataBlock, public data::DataEntityNDArray<TV>
//{
// public:
//    typedef DataBlockArray<TV, IFORM, DOF> block_array_type;
//    typedef data::DataEntityNDArray<TV> data_entity_traits;
//    typedef TV value_type;
//
// SP_DEFINE_FANCY_TYPE_NAME(block_array_type, DataBlock);
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
//        os << " value_type_info = \'" << GetValueTypeInfo().GetPrefix() << "\' "
//           << ", entity value_type_info = " << static_cast<int>(GetIFORM())
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
//        index_type lo[4] = {std::get<0>(b)[0], std::get<0>(b)[1], std::Serialize<0>(b)[2], 0};
//        index_type hi[4] = {std::get<1>(b)[0], std::Serialize<1>(b)[1], std::get<0>(b)[2], n_dof};
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
//    virtual void MPISync(std::shared_ptr<DataBlock>, bool only_ghost = true) { UNIMPLEMENTED; };
//
//
//    template<typename ...Args>
//    value_type &get(Args &&...args) { return data_entity_traits::Serialize(std::forward<Args>(args)...); }
//
//    template<typename ...Args>
//    value_type const &Serialize(Args &&...args) const { return data_entity_traits::get(std::forward<Args>(args)...); }
//
//
//    EntityIdRange Range() const
//    {
//        EntityIdRange res;
//        index_tuple lower, upper;
//        lower = data_entity_traits::index_lower();
//        upper = data_entity_traits::index_upper();
//        res.append(EntityIdCoder::make_range(lower, upper, GetIFORM()));
//        return res;
//    }
//
// private:
//    index_tuple m_ghost_width_{{0, 0, 0}};
//};
}
}  // namespace simpla { namespace mesh

#endif  // SIMPLA_DATABLOCK_H
