//
// Created by salmon on 16-6-6.
//
#include "DataEntity.h"
#include "DataTable.h"
namespace simpla {
namespace data {
DataEntity::DataEntity() {}
DataEntity::~DataEntity() {}

// DataEntity DataEntity::operator[](std::string const& url) {
//    if (m_data_ == nullptr) {
//        m_data_ = new DataTable;
//    } else if (!m_data_->isA(typeid(DataTable))) {
//        RUNTIME_ERROR << "Data entity is not indexable!" << std::endl;
//    }
//    return static_cast<DataTable*>(m_data_)->Insert(url).first;
//};
// DataEntity DataEntity::operator[](std::string const& url) const {
//    if (!m_data_->isA(typeid(DataTable))) { RUNTIME_ERROR << "Data entity is not indexable!" << std::endl; }
//    return (*static_cast<DataTable const*>(m_data_))[url];
//}
}  // namespace get_mesh{
}  // namespace simpla{