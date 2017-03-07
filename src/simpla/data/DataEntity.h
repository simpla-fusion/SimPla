//
// Created by salmon on 16-10-28.
//

#ifndef SIMPLA_DATAENTITY_H
#define SIMPLA_DATAENTITY_H
#include <simpla/concept/Printable.h>
#include <simpla/engine/SPObjectHead.h>
#include <simpla/toolbox/Log.h>
#include <typeindex>
#include <vector>
namespace simpla {
namespace data {
/** @ingroup data */

/**
 * @brief primary object of data
 */
struct DataHolderBase {
    SP_OBJECT_BASE(DataHolderBase);

   public:
    DataHolderBase() {}
    virtual ~DataHolderBase() {}
    virtual std::ostream& Print(std::ostream& os, int indent = 0) const = 0;
    virtual bool empty() const = 0;
    virtual DataHolderBase* Copy() const = 0;
    virtual std::type_info const& type() = 0;
};
template <typename U, typename Enable = void>
struct DataHolder : public DataHolderBase {
    SP_OBJECT_HEAD(DataHolder<U>, DataHolderBase);

   public:
    DataHolder() {}
    DataHolder(U const& d) : m_value_(d) {}
    DataHolder(U&& d) : m_value_(d) {}
    ~DataHolder() {}
    std::type_info const& type() { return typeid(U); };
    std::type_info const& type() const { return typeid(U); };

    std::ostream& Print(std::ostream& os, int indent = 0) const {
        os << m_value_;
        return os;
    };
    this_type& operator=(U const& v) {
        m_value_ = v;
        return *this;
    }
    virtual bool empty() const { return false; }
    virtual DataHolderBase* Copy() const { return new DataHolder<U>(m_value_); };
    virtual bool equal(U const& other) const { return other == m_value_; }

    virtual U value() const { return m_value_; }
    virtual U const* pointer() const { return &m_value_; }
    virtual U* pointer() { return &m_value_; }

   private:
    U m_value_;
};

struct DataEntity : public concept::Printable {
    SP_OBJECT_BASE(DataEntity);
    DataHolderBase* m_data_ = nullptr;

   public:
    DataEntity(DataHolderBase* p = nullptr);
    DataEntity(DataEntity const& other);
    DataEntity(DataEntity&& other);
    virtual ~DataEntity();

    template <typename U>
    DataEntity(U const& u) : m_data_(new DataHolder<U>(u)){};
    template <typename U>
    DataEntity(U&& u) : m_data_(new DataHolder<U>(u)){};

    std::type_info const& type() const;
    void swap(DataEntity& other);
    DataEntity& operator=(DataEntity const& other);
    bool empty() const;
    std::ostream& Print(std::ostream& os, int indent = 0) const;

    DataEntity& operator[](std::string const& key);
    DataEntity const& operator[](std::string const& key) const;

    template <typename U>
    DataEntity& operator=(U const& other) {
        DataEntity(other).swap(*this);
        return *this;
    }
    template <typename U>
    DataEntity& operator=(U&& other) {
        DataEntity(other).swap(*this);
        return *this;
    }

    template <typename U>
    bool operator==(U const& v) const {
        return (m_data_ != nullptr) && (m_data_->isA(typeid(U))) && static_cast<DataHolder<U> const*>(this)->equal(v);
    }

    template <typename U>
    U GetValue() const {
        if (type() != typeid(U)) { THROW_EXCEPTION_BAD_CAST(type().name(), typeid(U).name()); }
        return static_cast<DataHolder<U> const*>(m_data_)->value();
    }

    template <typename U>
    U GetValue(U const& default_value) const {
        if (type() != typeid(U)) { THROW_EXCEPTION_BAD_CAST(type().name(), typeid(U).name()); }

        return static_cast<DataHolder<U> const*>(m_data_)->value();
    }
};

template <typename U>
DataEntity make_data_entity(U const& u) {
    return DataEntity(u);
}
template <typename U>
DataEntity make_data_entity(U&& u) {
    return DataEntity(std::forward<U>(u));
}
inline DataEntity make_data_entity(char const* u) { return DataEntity(std::string(u)); }

}  // namespace data {
}  // namespace simpla {
#endif  // SIMPLA_DATAENTITY_H
