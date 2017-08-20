//
// Created by salmon on 17-8-20.
//

#ifndef SIMPLA_DATALIGHT_H
#define SIMPLA_DATALIGHT_H

#include "DataEntity.h"
namespace simpla {
namespace data {

struct DataLight : public DataEntity {
    SP_OBJECT_HEAD(DataLight, DataEntity);

   protected:
    DataLight() = default;

   public:
    ~DataLight() override = default;
    //    SP_DEFAULT_CONSTRUCT(DataLight);

    template <typename U>
    static std::shared_ptr<this_type> New(U const& u);

    template <typename U>
    bool Check(U const& u) const;
    template <typename U>
    U as() const;
    std::experimental::any any() const override;
};

template <typename V>
class DataLightT<V> : public DataLight {
    SP_OBJECT_HEAD(DataLightT, DataLight);
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
    value_type value() const { return m_data_; };

    std::type_info const& value_type_info() const override { return typeid(V); };
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

template <typename V>
class DataLightNTuple<V> : public DataLight {
    SP_OBJECT_HEAD(DataLightNTuple, DataLight);
    typedef V value_type;
    value_type m_data_;

   protected:
    DataLightNTuple() = default;
    template <typename... Args>
    explicit DataLightNTuple(Args&&... args) : m_data_(std::forward<Args>(args)...) {}

   public:
    ~DataLightNTuple() override = default;
    //    SP_DEFAULT_CONSTRUCT(DataLightNTuple);

    template <typename... Args>
    static std::shared_ptr<this_type> New(Args&&... args) {
        return std::shared_ptr<this_type>(new this_type(std::forward<Args>(args)...));
    }
    value_type value() const { return m_data_; };

    std::type_info const& value_type_info() const override { return typeid(V); };
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

template <typename U>
U DataLight::as() const {
    auto p = dynamic_cast<DataLightT<U> const*>(this);
    if (p == nullptr) { BAD_CAST; }
    return p->value();
}
template <typename U>
std::shared_ptr<DataLight> DataLight::New(U const& u) {
    return DataLightT<U>::New(u);
}

}  // namespace data
}  // namespace simpla
#endif  // SIMPLA_DATALIGHT_H
