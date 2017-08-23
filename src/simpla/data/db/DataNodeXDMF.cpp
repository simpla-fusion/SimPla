//
// Created by salmon on 17-8-13.
//
#include "DataNodeXDMF.h"
#include <sys/stat.h>
#include <Xdmf.hpp>
#include <XdmfDomain.hpp>
#include <XdmfWriter.hpp>

#include "../DataNode.h"

namespace simpla {
namespace data {
REGISTER_CREATOR(DataNodeXDMF, xmf);

//
//// int DataBackendXDMFRoot::Connect(std::string const& authority, std::string const& path, std::string const& query,
////                                 std::string const& fragment) {
////    m_prefix_ = path;
////}
//#define NUMBER_OF_FILE_COUNT_DIGTALS 6
//
// std::string add_count_suffix(std::string const& prefix, size_type n, int width = NUMBER_OF_FILE_COUNT_DIGTALS) {
//    std::ostringstream os;
//    os << prefix.c_str() << "." << std::setw(width) << std::setfill('0') << n;
//    return os.str();
//}
// void DataBackendXDMFRoot::Initialize() {
//    if (!m_pwd_.empty()) { return; }
//    Flush();
//    m_pwd_ = add_count_suffix(m_prefix_, m_counter_);
//    m_local_prefix_ = add_count_suffix(m_prefix_ + "/summary", m_local_rank_);
//    auto error_code = mkdir(m_pwd_.c_str(), 0777);
//    if (error_code != SP_SUCCESS) { RUNTIME_ERROR << "Can not make directory [" << m_pwd_ << "]" << std::endl; }
//
//    m_root_ = XdmfDomain::New();
//}
// int DataBackendXDMFRoot::Flush() {
//    if (m_root_ != nullptr) { m_root_->accept(XdmfWriter::New(m_local_prefix_ + ".xmf")); }
//    m_pwd_ = "";
//    ++m_counter_;
//    return 0;
//}
struct DataNodeXDMF::pimpl_s {
    std::string m_name_;
    std::shared_ptr<DataNodeXDMF> m_parent_ = nullptr;
    std::shared_ptr<XdmfItem> m_self_;
};
DataNodeXDMF::DataNodeXDMF() : m_pimpl_(new pimpl_s) {}
DataNodeXDMF::DataNodeXDMF(pimpl_s* pimpl) : m_pimpl_(pimpl) {}
DataNodeXDMF::~DataNodeXDMF() { delete m_pimpl_; }
bool DataNodeXDMF::isValid() const { return true; }
int DataNodeXDMF::Connect(std::string const& authority, std::string const& path, std::string const& query,
                          std::string const& fragment) {
    //    m_pimpl_->m_parent_->Connect(authority, path, query, fragment);
    return SP_SUCCESS;
}

int DataNodeXDMF::Disconnect() { return SP_SUCCESS; }

int DataNodeXDMF::Flush() { return 0; }

std::shared_ptr<DataNode> DataNodeXDMF::Duplicate() const {}
size_type DataNodeXDMF::GetNumberOfChildren() const {}
DataNode::e_NodeType DataNodeXDMF::NodeType() const {}
std::shared_ptr<DataNode> DataNodeXDMF::Root() const {
    return Parent() != nullptr ? Parent()->Root() : const_cast<this_type*>(this)->shared_from_this();
}
std::shared_ptr<DataNode> DataNodeXDMF::Parent() const { return m_pimpl_->m_parent_; }

int DataNodeXDMF::Foreach(std::function<int(std::string, std::shared_ptr<DataNode>)> const& fun) {}
int DataNodeXDMF::Foreach(std::function<int(std::string, std::shared_ptr<DataNode>)> const& fun) const {}

std::shared_ptr<DataNode> DataNodeXDMF::GetNode(std::string const& uri, int flag) {}
std::shared_ptr<DataNode> DataNodeXDMF::GetNode(std::string const& uri, int flag) const {}
std::shared_ptr<DataNode> DataNodeXDMF::GetNode(index_type s, int flag) {}
std::shared_ptr<DataNode> DataNodeXDMF::GetNode(index_type s, int flag) const {}
int DataNodeXDMF::DeleteNode(std::string const& uri, int flag) {}
void DataNodeXDMF::Clear() {}

std::shared_ptr<DataEntity> DataNodeXDMF::Get() {}
std::shared_ptr<DataEntity> DataNodeXDMF::Get() const {}
int DataNodeXDMF::Set(std::shared_ptr<DataEntity> const& v) {}
int DataNodeXDMF::Add(std::shared_ptr<DataEntity> const& v) {}

////
//// std::shared_ptr<DataEntity> DataBaseXDMF::Get(std::string const& URI) const {
////    if (URI[0] == '/') {
////        //        GetRoot()->Get(URI.substr(1));
////    } else {
////        auto new_backend = DataBaseXDMF::New();
////        new_backend->m_pimpl_->m_parent_ =
////            std::dynamic_pointer_cast<this_type>(const_cast<this_type*>(this)->shared_from_this());
////        new_backend->m_pimpl_->m_name_ = URI;
////        auto res = DataTable::New(new_backend);
////    }
////    return nullptr;
////}
//// int DataBaseXDMF::Set(std::string const& URI, const std::shared_ptr<DataEntity>& entity) { return 0; }
//// int DataBaseXDMF::Add(std::string const& URI, const std::shared_ptr<DataEntity>& entity) { return 0; }
//// int DataBaseXDMF::Delete(std::string const& URI) { return 0; }
//// int DataBaseXDMF::Foreach(std::function<int(std::string const&, std::shared_ptr<DataEntity>)> const& fun) const {
////    return 0;
////}

}  // namespace data{
}  // namespace simpla