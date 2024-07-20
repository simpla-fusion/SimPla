//
// Created by salmon on 16-11-2.
//

#ifndef SIMPLA_DATABLOCK_H
#define SIMPLA_DATABLOCK_H
#include <memory>
#include <typeinfo>

#include "simpla/config.h"
namespace simpla {
/**
 *  Base class of Data Blocks (pure virtual)
 */

template <typename TValue>
class DataBlock {
   public:
    typedef TValue ValueType;
    typedef DataBlock<TValue> ThisType;

    explicit DataBlock() = default;

    template <typename... Args>
    explicit DataBlock(Args &&...args){};

   public:
    ~DataBlock() = default;

    template <typename... Args>
    decltype(auto) get_value(Args &&...args) const {
        return nullptr;
    }

    template <typename... Args>
    decltype(auto) set_value(Args &&...args) {}

    // template <typename... Args>
    // static std::shared_ptr<ThisType> create(Args &&...args);

    // size_type size() const override { return 0; };

    // int GetNDIMS() const override { return array_type::GetNDIMS(); };
    // int GetIndexBox(index_type *lo, index_type *hi) const override { return array_type::GetIndexBox(lo, hi); };

    // void *GetPointer() override { return Array<V>::get(); }
    // void const *GetPointer() const override { return Array<V>::get(); }

    // void clear() override { array_type::clear(); };

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
    // std::string __str__(std::ostream &os, int indent) const override {
    //     // std::ostream os
    //     // os << "Array<" << simpla::traits::type_name<ValueType>::value() << ">" <<
    //     // Array<ValueType>::GetIndexBox();
    //     return os.str();
    // }
};
// template <typename U>
// std::shared_ptr<DataBlock<U>> make_data_block(U *p, int rank, index_type const *lo, index_type const *hi) {
//     return DataBlock<U>::New(p, rank, lo, hi);
// }

// template <typename U>
// std::shared_ptr<DataBlock<U>> make_data_block(std::shared_ptr<U> const &p, int rank, index_type const *lo,
//                                               index_type const *hi) {
//     return DataBlock<U>::New(p, rank, lo, hi);
// }
// template <typename U>
// std::shared_ptr<DataBlock<U>> make_data_block(std::shared_ptr<U> const &p, index_box_type const &ibox) {
//     return DataBlock<U>::New(p, 3, &std::get<0>(ibox)[0], &std::get<1>(ibox)[0]);
// }
// template <typename V>
// template <typename... Args>
// std::shared_ptr<DataBlock<V>> DataBlock<V>::New(Args &&...args) {
//     return std::shared_ptr<DataBlock<V>>(new DataBlock<V>(std::forward<Args>(args)...));
// };
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
//    typedef U ValueType;
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
//    std::type_info const &ValueType_info() const override { return typeid(ValueType); };
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
//    void DeepCopy(std::shared_ptr<ThisType> const &other) {
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
//   *    virtual void load(DataTable const &) =0;
//   *    virtual void save(DataTable *) const =0;
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
//    typedef algebra::traits::ValueType_t<U> ValueType;
//
//   public:
//    template <typename... Args>
//    explicit DataBlockAdapter(Args &&... args) : U(std::forward<Args>(args)...) {}
//    ~DataBlockAdapter() {}
//    virtual std::type_info const &ValueType_info() const { return typeid(algebra::traits::ValueType_t<U>); };
//    virtual int entity_type() const { return algebra::traits::iform<U>::value; }
//    virtual int dof() const { return algebra::traits::dof<U>::value; }
//    virtual std::ostream &Print(std::ostream &os, int indent) const {
//        os << " ValueType_info = \'" << ValueType_info().name() << "\' "
//           << ", entity ValueType_info = " << (entity_type()) << ", GetDOF = " << (dof()) << ", GetDataBlock = {";
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
// class DataBlockArray : public DataBlock, public DataEntityNDArray<TV>
//{
// public:
//    typedef DataBlockArray<TV, IFORM, DOF> block_array_type;
//    typedef DataEntityNDArray<TV> data_entity_traits;
//    typedef TV ValueType;
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
//    virtual std::type_info const &GetValueTypeInfo() const { return typeid(ValueType); };
//
//    virtual int GetIFORM() const { return IFORM; }
//
//    virtual int GetDOF() const { return DOF; }
//
//    virtual void Load(DataTable const &) { UNIMPLEMENTED; };
//
//    virtual void Save(DataTable *) const { UNIMPLEMENTED; };
//
//    virtual std::ostream &Print(std::ostream &os, int indent) const
//    {
//        os << " ValueType_info = \'" << GetValueTypeInfo().GetPrefix() << "\' "
//           << ", entity ValueType_info = " << static_cast<int>(GetIFORM())
//           << ", GetDataBlock = {";
//        data_entity_traits::Print(os, indent + 1);
//        os << "}";
//        return os;
//    }
//
//    virtual std::shared_ptr<DataBlock> clone(std::shared_ptr<RectMesh> const &m, void *p = nullptr)
//    {
//        return create(m, static_cast<ValueType *>(p));
//    };
//
//
//    static std::shared_ptr<DataBlock>
//    create(std::shared_ptr<RectMesh> const &m, ValueType *p = nullptr)
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
//        return std::dynamic_pointer_cast<DataBlock>(std::make_shared<ThisType>(p, ndims, lo, hi));
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
//    ValueType &get(Args &&...args) { return data_entity_traits::Serialize(std::forward<Args>(args)...); }
//
//    template<typename ...Args>
//    ValueType const &Serialize(Args &&...args) const { return data_entity_traits::get(std::forward<Args>(args)...); }
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

}  // namespace simpla

#endif  // SIMPLA_DATABLOCK_H
