//
// Created by salmon on 17-3-8.
//

#ifndef SIMPLA_DATAARRAY_H
#define SIMPLA_DATAARRAY_H
#include "DataEntity.h"
#include "DataTraits.h"
namespace simpla {
namespace data {
template <typename U, typename Enable = void>
class DataArrayWrapper {};
struct DataArray : public DataEntity {
    SP_OBJECT_HEAD(DataArray, DataEntity)
   public:
    DataArray();
    virtual ~DataArray();
    virtual std::ostream& Print(std::ostream& os, int indent = 0) const;
    virtual bool isArray() const { return true; }
    /** as Array */
    virtual size_type Count() const = 0;  // { return 0; };
    virtual std::shared_ptr<DataEntity> Get(size_type idx) const { return std::make_shared<DataEntity>(); }
    virtual bool Set(size_type idx, std::shared_ptr<DataEntity> const&) { return false; }
    virtual bool Add(std::shared_ptr<DataEntity> const&) { return false; }
    virtual int Delete(size_type idx) { return 0; }
};
template <>
struct DataArrayWrapper<void> : public DataArray {
    SP_OBJECT_HEAD(DataArrayWrapper<void>, DataArray)
   public:
    DataArrayWrapper() {}

    template <typename U>
    DataArrayWrapper(std::initializer_list<U> const& v) {
        for (auto const& item : v) { m_data_.push_back(make_data_entity(v)); }
    };

    virtual ~DataArrayWrapper(){};

    virtual size_type Count() const { return m_data_.size(); }
    virtual std::shared_ptr<DataEntity> Get(size_type idx) const { return m_data_[idx]; }
    virtual bool Set(size_type idx, std::shared_ptr<DataEntity> const& v) {
        if (idx < size()) {
            m_data_[idx] = v;
            return true;
        } else {
            return false;
        }
    }
    virtual bool Add(std::shared_ptr<DataEntity> const& v) {
        m_data_.push_back(v);
        return true;
    }
    virtual int Delete(size_type idx) {
        m_data_.erase(m_data_.begin() + idx);
        return 1;
    }

   private:
    std::vector<std::shared_ptr<DataEntity>> m_data_;
};

template <typename U>
class DataArrayWrapper<U, std::enable_if_t<traits::is_light_data<U>::value>> : public DataArray {
    SP_OBJECT_HEAD(DataArrayWrapper<U>, DataArray);
    std::vector<U> m_data_;

   public:
    DataArrayWrapper() {}
    DataArrayWrapper(this_type const& other) : m_data_(other.m_data_) {}
    DataArrayWrapper(this_type&& other) : m_data_(other.m_data_) {}

    virtual ~DataArrayWrapper() {}
    virtual std::type_info const& type() const { return typeid(U); };

    virtual size_type Count() const { return m_data_.size(); };
    virtual std::shared_ptr<DataEntity> Get(size_type idx) const { return make_data_entity(m_data_[idx]); }

    virtual bool Set(size_type idx, U const& v) {
        if (size() > idx) {
            m_data_[idx] = v;
            return true;
        } else {
            return false;
        }
    }
    virtual bool Set(size_type idx, std::shared_ptr<DataEntity> const& v) { return Set(idx, v->as<U>()); }

    virtual bool Add(U const& v) {
        m_data_.push_back(v);
        return true;
    }
    virtual bool Add(std::shared_ptr<DataEntity> const& v) { return Add(v->as<U>()); }
    virtual int Delete(size_type idx) {
        m_data_.erase(m_data_.begin() + idx);
        return 1;
    }
};

template <typename U>
std::shared_ptr<DataEntity> make_data_entity(std::initializer_list<U> const& u,
                                             ENABLE_IF(traits::is_light_data<U>::value)) {
    auto res = std::make_shared<DataArrayWrapper<U>>();
    for (U const& v : u) { res->Add(v); }
    return std::dynamic_pointer_cast<DataEntity>(res);
}

inline std::shared_ptr<DataEntity> make_data_entity(std::initializer_list<char const*> const& u) {
    auto res = std::make_shared<DataArrayWrapper<std::string>>();
    for (char const* v : u) { res->Add(v); }
    return std::dynamic_pointer_cast<DataEntity>(res);
}

template <typename U>
std::shared_ptr<DataEntity> make_data_entity(std::initializer_list<U> const& u,
                                             ENABLE_IF(!traits::is_light_data<U>::value)) {
    auto res = std::make_shared<DataArrayWrapper<void>>();
    for (auto const& v : u) { res->Add(make_data_entity(v)); }
    return std::dynamic_pointer_cast<DataEntity>(res);
}
template <typename U>
std::shared_ptr<DataEntity> make_data_entity(std::initializer_list<std::initializer_list<U>> const& u) {
    auto res = std::make_shared<DataArrayWrapper<void>>();
    for (auto const& v : u) { res->Add(make_data_entity(v)); }
    return std::dynamic_pointer_cast<DataEntity>(res);
}
}  // namespace data{
}  // namespace simpla{
#endif  // SIMPLA_DATAARRAY_H
