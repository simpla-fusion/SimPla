//
// Created by salmon on 17-8-20.
//

#ifndef SIMPLA_DATALIGHT_H
#define SIMPLA_DATALIGHT_H

#include "DataEntity.h"
#include "simpla/utilities/FancyStream.h"
namespace simpla {
namespace data {

struct DataLight : public DataEntity {
    SP_DEFINE_FANCY_TYPE_NAME(DataLight, DataEntity);

   protected:
    DataLight() = default;

   public:
    ~DataLight() override = default;
    static std::shared_ptr<DataLight> New();

    virtual void* GetPointer() { return nullptr; }
    virtual void const* GetPointer() const { return nullptr; }
};
template <typename V>
class DataLightT : public DataLight {
    SP_DEFINE_FANCY_TYPE_NAME(DataLightT, DataLight);
    typedef V value_type;
    value_type m_data_;

   protected:
    DataLightT() = default;

    template <typename... Args>
    explicit DataLightT(Args&&... args) : m_data_(std::forward<Args>(args)...) {}

   public:
    ~DataLightT() override = default;
    //    SP_DEFAULT_CONSTRUCT(DataLightT);
    template <typename... Args>
    static std::shared_ptr<this_type> New(Args&&... args) {
        return std::shared_ptr<this_type>(new this_type(std::forward<Args>(args)...));
    }

    bool equal(DataEntity const& other) const override {
        auto* p = dynamic_cast<DataLightT<value_type> const*>(&other);
        return p != nullptr && (p->m_data_ == m_data_);
    }
    virtual bool value_equal(void const* other, std::type_info const& info) const override {
        return value_type_info() == info && m_data_ == *reinterpret_cast<value_type const*>(other);
    }
    std::ostream& Print(std::ostream& os, int indent) const override {
        if (typeid(value_type) == typeid(std::string)) {
            os << "\"" << m_data_ << "\"";
        } else {
            os << m_data_;
        }
        return os;
    }

    value_type const& value() const { return m_data_; };
    value_type& value() { return m_data_; };
    void* GetPointer() override { return reinterpret_cast<void*>(&m_data_); }
    void const* GetPointer() const override { return reinterpret_cast<void const*>(&m_data_); }
    std::type_info const& value_type_info() const override { return typeid(value_type); };
    size_type value_type_size() const override { return sizeof(value_type); };
    size_type rank() const override { return std::rank<value_type>::value; }
    size_type extents(size_type* d) const override {
        if (d != nullptr) {
            switch (rank() - 1) {
                default:
                    UNIMPLEMENTED;
                case 9:
                    d[9] = std::extent<value_type, 9>::value;
                case 8:
                    d[8] = std::extent<value_type, 8>::value;
                case 7:
                    d[7] = std::extent<value_type, 7>::value;
                case 6:
                    d[6] = std::extent<value_type, 6>::value;
                case 5:
                    d[5] = std::extent<value_type, 5>::value;
                case 4:
                    d[4] = std::extent<value_type, 4>::value;
                case 3:
                    d[3] = std::extent<value_type, 3>::value;
                case 2:
                    d[2] = std::extent<value_type, 2>::value;
                case 1:
                    d[1] = std::extent<value_type, 1>::value;
                case 0:
                    d[0] = std::extent<value_type, 0>::value;
                    break;
            }
        };
        return rank();
    }
    size_type size() const override {
        size_type res = 1;
        size_type d[10];
        extents(d);
        for (int i = 0; i < rank(); ++i) { res *= d[i]; }
        return res;
    }

    std::experimental::any any() const override { return std::experimental::any(m_data_); };
};

}  // namespace data
}  // namespace simpla
#endif  // SIMPLA_DATALIGHT_H
