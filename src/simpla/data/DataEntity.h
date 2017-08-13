//
// Created by salmon on 16-10-28.
//

#ifndef SIMPLA_DATAENTITY_H
#define SIMPLA_DATAENTITY_H
#include "simpla/SIMPLA_config.h"

#include <typeindex>
#include <vector>
#include "DataTraits.h"
#include "simpla/utilities/Log.h"
#include "simpla/utilities/ObjectHead.h"
namespace simpla {
namespace data {
template <typename, typename Enable = void>
class DataEntityWrapper {};
struct DataEntity {
    SP_OBJECT_BASE(DataEntity);

   public:
    DataEntity() = default;
    virtual ~DataEntity() = default;

    SP_DEFAULT_CONSTRUCT(DataEntity)

    virtual std::shared_ptr<DataEntity> Duplicate() const = 0;
    virtual std::ostream& Serialize(std::ostream& os, int indent) const;
    virtual std::istream& Deserialize(std::istream& is);

    virtual bool empty() const { return true; }
    virtual std::type_info const& value_type_info() const { return typeid(void); };
    virtual bool isLight() const { return false; }
    virtual bool isNull() const;

    template <typename U>
    operator U() const {
        return dynamic_cast<DataEntityWrapper<U> const*>(this)->value();
    }
};
inline std::ostream& operator<<(std::ostream& os, DataEntity const& v) {
    v.Serialize(os, 0);
    return os;
}
inline std::istream& operator<<(std::istream& is, DataEntity& v) {
    v.Deserialize(is);
    return is;
}

template <typename V>
struct DataEntityWrapper<V> : public DataEntity {
    SP_OBJECT_HEAD(DataEntityWrapper<V>, DataEntity);
    typedef V value_type;
    value_type* m_data_ = nullptr;
    bool m_owen_ = true;

   public:
    DataEntityWrapper() = default;
    ~DataEntityWrapper() override {
        if (m_owen_) { delete m_data_; }
    };
    explicit DataEntityWrapper(value_type const& d) : m_data_(new V(d)), m_owen_(true) {}
    explicit DataEntityWrapper(value_type const* d) : m_data_(new V(*d)), m_owen_(true) {}
    explicit DataEntityWrapper(value_type* d) : m_data_(d), m_owen_(false) {}
    DataEntityWrapper(const this_type& other)
        : m_data_((other.m_owen_) ? (new value_type(other.value())) : other.m_data_), m_owen_(other.m_owen_){};
    DataEntityWrapper(this_type&& other) noexcept : m_data_(other.m_data_), m_owen_(other.m_owen_) {
        other.m_data_ = nullptr;
        other.m_owen_ = false;
    };
    template <typename U>
    DataEntityWrapper& operator=(const U& other) {
        *m_data_ = static_cast<value_type>(other);
        return *this;
    };
    std::shared_ptr<DataEntity> Duplicate() const override {
        return std::dynamic_pointer_cast<DataEntity>(std::make_shared<this_type>(*this));
    }

    std::ostream& Serialize(std::ostream& os, int indent) const override {
        os << *m_data_;
        return os;
    }

    std::type_info const& value_type_info() const override { return typeid(value_type); }
    bool isLight() const override { return true; }
    value_type& value() { return *m_data_; };
    value_type const& value() const { return *m_data_; };
    value_type* get() { return m_data_; }
};

template <>
struct DataEntityWrapper<std::string> : public DataEntity {
    SP_OBJECT_HEAD(DataEntityWrapper<std::string>, DataEntity);
    typedef std::string value_type;
    value_type m_data_;

   public:
    DataEntityWrapper() = default;
    ~DataEntityWrapper() override{};
    explicit DataEntityWrapper(value_type const& s) : m_data_(s) {}
    explicit DataEntityWrapper(value_type const* s) : m_data_(*s) {}

    DataEntityWrapper(const this_type& other) : m_data_(other.m_data_){};
    DataEntityWrapper(this_type&& other) noexcept : m_data_(other.m_data_){};
    template <typename U>
    DataEntityWrapper& operator=(const U& other) {
        m_data_ = static_cast<value_type>(other);
        return *this;
    };
    std::shared_ptr<DataEntity> Duplicate() const override {
        return std::dynamic_pointer_cast<DataEntity>(std::make_shared<this_type>(*this));
    }

    std::ostream& Serialize(std::ostream& os, int indent) const override {
        os << m_data_;
        return os;
    }

    std::type_info const& value_type_info() const override { return typeid(value_type); }
    bool isLight() const override { return true; }
    value_type& value() { return m_data_; };
    value_type const& value() const { return m_data_; };
    char const* get() const { return m_data_.c_str(); }
};
template <typename U>
std::shared_ptr<DataEntityWrapper<U>> make_data_entity(U const& u) {
    return std::make_shared<DataEntityWrapper<U>>(u);
}
inline std::shared_ptr<DataEntityWrapper<std::string>> make_data_entity(char const* c) {
    return std::make_shared<DataEntityWrapper<std::string>>(std::string(c));
}

template <typename U>
U data_cast(DataEntity const& p) {
    return dynamic_cast<DataEntityWrapper<U> const&>(p).value();
}
template <typename U>
U data_cast(DataEntity const& p, U const& default_value) {
    auto const* tp = dynamic_cast<DataEntityWrapper<U> const*>(&p);
    return (tp == nullptr) ? default_value : tp->value();
}

}  // namespace data {
}  // namespace simpla {
#endif  // SIMPLA_DATAENTITY_H
