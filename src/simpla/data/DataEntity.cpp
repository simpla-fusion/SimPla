//
// Created by salmon on 16-6-6.
//
#include "DataEntity.h"
#include "DataBlock.h"
#include "DataTable.h"

namespace simpla {
namespace data {
// DataEntity::DataEntity() {}
// DataEntity::~DataEntity() {}

bool DataEntity::isNull() const {
    return dynamic_cast<DataTable const*>(this) == nullptr &&  //
           dynamic_cast<DataArray const*>(this) == nullptr &&  //
           dynamic_cast<DataBlock const*>(this) == nullptr &&  //
           (!isLight());
}

std::ostream& DataEntity::Serialize(std::ostream& os, int indent) const {
    if (isLight()) {
        os << "<Light Data:" << value_type_info().name() << ">";
    } else if (dynamic_cast<DataBlock const*>(this) != nullptr) {
        os << "<Block:" << value_type_info().name() << "," << std::boolalpha
           << dynamic_cast<DataBlock const*>(this)->empty() << ">";
    } else if (dynamic_cast<DataTable const*>(this) != nullptr) {
        os << "<Table:" << value_type_info().name() << ">";
    } else if (dynamic_cast<DataArray const*>(this) != nullptr) {
        os << "<Array:" << value_type_info().name() << ">";
    }
    return os;
};
std::istream& DataEntity::Deserialize(std::istream& is) { return is; }
// DataEntity DataEntity::operator[](std::string const& url) {
//    if (m_holder_ == nullptr) {
//        m_holder_ = new DataTable;
//    } else if (!m_holder_->isA(typeid(DataTable))) {
//        RUNTIME_ERROR << "Data entity is not indexable!" << std::endl;
//    }
//    return static_cast<DataTable*>(m_holder_)->Insert(url).first;
//};
// DataEntity DataEntity::operator[](std::string const& url) const {
//    if (!m_holder_->isA(typeid(DataTable))) { RUNTIME_ERROR << "Data entity is not indexable!" << std::endl; }
//    return (*static_cast<DataTable const*>(m_holder_))[url];
//}
}  // namespace get_mesh{
}  // namespace simpla{