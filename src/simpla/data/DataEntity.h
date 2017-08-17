//
// Created by salmon on 16-10-28.
//

#ifndef SIMPLA_DATAENTITY_H
#define SIMPLA_DATAENTITY_H
#include "simpla/SIMPLA_config.h"

#include <simpla/data/db/DataBaseStdIO.h>
#include <typeindex>
#include <vector>
#include "DataEntityVisitor.h"
#include "DataTraits.h"
#include "simpla/utilities/Log.h"
#include "simpla/utilities/ObjectHead.h"
namespace simpla {
namespace data {
template <typename, typename Enable = void>
class DataEntityWrapper {};

enum DataEntityType { DB_NULL = 0, DB_LIGHT = 1, DB_BLOCK = 2, DB_ARRAY = 3, DB_TABLE = 4 };
class DataEntityVisitor;
struct DataEntity : public std::enable_shared_from_this<DataEntity> {
    SP_OBJECT_BASE(DataEntity);

   private:
    std::shared_ptr<DataEntity> m_parent_ = nullptr;

   protected:
    explicit DataEntity(std::shared_ptr<DataEntity> const& parent = nullptr);

   public:
    SP_DEFAULT_CONSTRUCT(DataEntity)
    virtual ~DataEntity() = default;

    template <typename U>
    static std::shared_ptr<DataEntity> New(U const& v);
    virtual std::type_info const& value_type_info() const { return typeid(void); };
    virtual int GetTypeId() const;
    virtual DataEntity* GetParent();
    virtual DataEntity const* GetParent() const;
    virtual DataEntity* GetRoot();
    virtual DataEntity const* GetRoot() const;

    virtual std::ostream& Serialize(std::ostream& os, int indent) const;
    virtual std::istream& Deserialize(std::istream& is);

    virtual int Accept(DataEntityVisitor&) const { return 0; };

    virtual bool isNull() const;
    virtual bool isBlock() const;
    virtual bool isTable() const;
    virtual bool isArray() const;
    virtual bool isLight() const;
    virtual size_type Count() const { return 0; }

    template <typename U>
    bool Check(U const& u = true) const {
        auto p = dynamic_cast<DataEntityWrapper<U> const*>(this);
        return (p != nullptr) && p->value() == u;
    }
    template <typename U>
    bool isA() const {
        return dynamic_cast<DataEntityWrapper<U> const*>(this) != nullptr || dynamic_cast<U const*>(this) != nullptr;
    }
    template <typename U>
    U as() const {
        auto const* p = dynamic_cast<DataEntityWrapper<U> const*>(this);
        if (p == nullptr) { BAD_CAST << "Can not convert to type[" << typeid(U).name() << "]" << std::endl; }
        return p->value();
    }

    template <typename U>
    U as(U const& default_value) const {
        auto p = dynamic_cast<DataEntityWrapper<U> const*>(this);
        return p == nullptr ? default_value : p->value();
    }
};
std::ostream& operator<<(std::ostream& os, DataEntity const& v);
std::istream& operator<<(std::istream& is, DataEntity& v);

template <typename V>
struct DataEntityWrapper<V> : public DataEntity {
    SP_OBJECT_HEAD(DataEntityWrapper<V>, DataEntity);
    typedef V value_type;
    value_type m_data_;

   protected:
    DataEntityWrapper() = default;
    explicit DataEntityWrapper(value_type const& d) : m_data_(d) {}

   public:
    ~DataEntityWrapper() override = default;

    static std::shared_ptr<this_type> New(value_type const& d) { return std::shared_ptr<this_type>(new this_type(d)); }

    std::type_info const& value_type_info() const override { return typeid(value_type); };

    int Accept(DataEntityVisitor& visitor) const override { return visitor.visit(m_data_); };

    bool isLight() const override { return true; }

    size_type Count() const override { return 1; }

    value_type value() const { return m_data_; };
};

template <typename U>
std::shared_ptr<DataEntity> DataEntity::New(U const& v) {
    return DataEntityWrapper<U>::New(v);
}
template <typename U>
std::shared_ptr<DataEntityWrapper<U>> make_data_entity(U const& u, ENABLE_IF(traits::is_light_data<U>::value)) {
    return DataEntityWrapper<U>::New(u);
}
inline std::shared_ptr<DataEntityWrapper<std::string>> make_data_entity(char const* u) {
    return DataEntityWrapper<std::string>::New(std::string(u));
}

}  // namespace data {
}  // namespace simpla {
#endif  // SIMPLA_DATAENTITY_H
