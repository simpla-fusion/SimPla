//
// Created by salmon on 17-3-8.
//
#include "DataArray.h"
#include <iomanip>
#include "DataTraits.h"
namespace simpla {
namespace data {

DataArray::DataArray(std::shared_ptr<DataEntity> const& parent) : DataEntity(parent) {}

std::shared_ptr<DataArray> DataArray::New(std::shared_ptr<DataEntity> const& parent) {
    return DataArrayDefault::New(parent);
};

int DataArray::Foreach(std::function<int(std::shared_ptr<DataEntity>)> const& fun) const {
    int res = 0;
    for (size_type i = 0, ie = Count(); i < ie; ++i) { res += fun(Get(i)); }
    return res;
};

// std::ostream& DataArray::Serialize(std::ostream& os, int indent) const {
//    size_type ie = Count();
//    os << "[";
//    Get(0)->Serialize(os, indent + 1);
//    for (size_type i = 1; i < ie; ++i) {
//        os << " , ";
//        Get(i)->Serialize(os, indent + 1);
//        //        if (i % 5 == 0) { os << std::endl; }
//    }
//    os << "]";
//    return os;
//};
DataArrayDefault::DataArrayDefault(std::shared_ptr<DataEntity> const& parent) : DataArray(parent) {}

std::shared_ptr<DataArrayDefault> DataArrayDefault::New(std::shared_ptr<DataEntity> const& parent) {
    return std::shared_ptr<DataArrayDefault>(new DataArrayDefault(parent));
};

size_type DataArrayDefault::Count() const { return m_data_.size(); };

size_type DataArrayDefault::Resize(size_type s) {
    m_data_.resize(s);
    return s;
};

std::shared_ptr<DataEntity> DataArrayDefault::Get(size_type idx) const { return m_data_.at(idx); }

int DataArrayDefault::Set(size_type idx, std::shared_ptr<DataEntity> const& v) {
    bool success = idx < m_data_.size();
    if (success) { m_data_.at(idx) = v; }
    return 1;
}
int DataArrayDefault::Add(std::shared_ptr<DataEntity> const& v) {
    m_data_.push_back(v);
    return 1;
}

int DataArrayDefault::Delete(size_type idx) {
    m_data_.erase(m_data_.begin() + idx);
    return 1;
}
// DataArray::DataArray() {}
// DataArray::~DataArray() {}

}  // namespace data {
}  // namespace simpla {