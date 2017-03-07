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
template <typename U, typename Enable = void>
struct DataHolder;

struct DataHolderBase {
    SP_OBJECT_BASE(DataHolderBase);

   public:
    DataHolderBase() {}
    virtual ~DataHolderBase() {}
    virtual std::ostream& Print(std::ostream& os, int indent = 0) const = 0;
    virtual bool empty() const = 0;
    virtual DataHolderBase* Copy() const = 0;
    virtual std::type_info const& type() const = 0;

    template <typename U>
    U as() const {
        return static_cast<DataHolder<U> const*>(this)->value();
    }

    template <typename U>
    bool operator==(U const& other) const {
        //        return static_cast<DataHolder<U> const*>(this)->equal(other);
        return false;
    }
};
template <typename U>
struct DataHolder<U> : public DataHolderBase {
    SP_OBJECT_HEAD(DataHolder<U>, DataHolderBase);

   public:
    DataHolder() {}
    DataHolder(U const& d) : m_value_(d) {}
    DataHolder(U&& d) : m_value_(std::forward<U>(d)) {}
    ~DataHolder() {}
    std::type_info const& type() const { return typeid(U); };

    std::ostream& Print(std::ostream& os, int indent = 0) const {
        //        os << m_value_;
        return os;
    };
    this_type& operator=(U const& v) {
        m_value_ = v;
        return *this;
    }
    virtual bool empty() const { return false; }
    virtual DataHolderBase* Copy() const { return new DataHolder<U>(m_value_); };
    virtual bool equal(U const& other) const { return false; /* other == m_value_;*/ }
    virtual U value() const { return m_value_; }

   private:
    U m_value_;
};

struct DataEntity : public concept::Printable {
    DataHolderBase* m_data_ = nullptr;

   public:
    DataEntity(DataHolderBase* p = nullptr);
    DataEntity(DataEntity const&);
    DataEntity(DataEntity&&);

    virtual ~DataEntity();

    //    template <typename U>
    //    DataEntity(U const& u) : m_data_(new DataHolder<U>(u)){};
    //    template <typename U>
    //    DataEntity(U&& u) : m_data_(new DataHolder<U>(std::forward<U>(u))){};

    void swap(DataEntity& other);
    DataEntity& operator=(DataEntity const& other);
    std::ostream& Print(std::ostream& os, int indent = 0) const;

    bool empty() const;
    std::type_info const& type() const;

    template <typename U>
    bool operator==(U const& v) const {
        return (!empty()) && (*m_data_ == v);
    }

    template <typename U>
    U as() const {
        ASSERT(!empty());
        return m_data_->as<U>();
    }

    template <typename U>
    U as(U const& default_value) const {
        return empty() ? default_value : as<U>();
    }
};

template <typename U>
DataEntity make_data_entity(U const& u) {
    return DataEntity(u);
}
template <typename U>
DataEntity make_data_entity(U&& u) {
    return std::move(DataEntity(new DataHolder<U>(u)));
}
inline DataEntity make_data_entity(char const* u) {
    return std::move(DataEntity(new DataHolder<std::string>(std::string(u))));
}

}  // namespace data {
}  // namespace simpla {
#endif  // SIMPLA_DATAENTITY_H
