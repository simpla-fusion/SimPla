//
// Created by salmon on 17-8-13.
//
#include <sys/stat.h>
#include <Xdmf.hpp>
#include <XdmfDomain.hpp>
#include <XdmfGridCollection.hpp>
#include <XdmfWriter.hpp>
#include "../DataNode.h"
#include "DataNodeMemory.h"

namespace simpla {
namespace data {
int XDMFDump(std::string url, std::shared_ptr<DataNode> const& v);

struct DataNodeXDMF : public DataNodeMemory {
    SP_DATA_NODE_HEAD(DataNodeXDMF, DataNodeMemory)

    int Connect(std::string const& authority, std::string const& path, std::string const& query,
                std::string const& fragment) override {
        m_file_ = path;
        return SP_SUCCESS;
    }
    int Disconnect() override { return Flush(); }
    bool isValid() const override { return true; }
    int Flush() override { return XDMFDump(m_file_, shared_from_this()); }

   private:
    std::string m_file_;
};
REGISTER_CREATOR(DataNodeXDMF, xmf);
DataNodeXDMF::DataNodeXDMF(DataNode::eNodeType etype) : base_type(etype) {}
DataNodeXDMF::~DataNodeXDMF() = default;

//************************************************************************************************
// size_type XDMFCreateOrOpenGroup(std::shared_ptr<XdmfItem>& self, std::string const& uri) {}
// size_type XDMFSet(std::shared_ptr<XdmfItem>& self, std::string const& uri, const std::shared_ptr<DataNode>& v) {}
//
// shared_ptr<XdmfDomain> XDMFFromAtlas(std::shared_ptr<DataNode> const& v) {
//    auto root = XdmfDomain::New();
//    // Dimensions
//    shared_ptr<XdmfArray> dimensions = XdmfArray::New();
//    dimensions->pushBack((unsigned int)10);
//    dimensions->pushBack((unsigned int)20);
//    dimensions->pushBack((unsigned int)30);
//    // Origin
//    shared_ptr<XdmfArray> origin = XdmfArray::New();
//    origin->pushBack((unsigned int)1);
//    origin->pushBack((unsigned int)2);
//    origin->pushBack((unsigned int)3);
//    // Brick Size
//    shared_ptr<XdmfArray> brick = XdmfArray::New();
//    brick->pushBack((unsigned int)1);
//    brick->pushBack((unsigned int)2);
//    brick->pushBack((unsigned int)3);
//    // Grid
//    shared_ptr<XdmfRegularGrid> g = XdmfRegularGrid::New(brick, dimensions, origin);
//    auto grid = XdmfGridCollection::New();
//
//    auto geometry = XdmfGeometry::New();
//    auto topology = XdmfTopology::New();
//}
// template <typename U, int N>
// std::shared_ptr<XdmfArray> XDMFArray(nTuple<U, N> const& v){};
int XDMFDump(std::string url, std::shared_ptr<DataNode> const& v) {
    if (v == nullptr) { return SP_FAILED; }
    auto domain = XdmfDomain::New();
    //    size_type count = 0;
    //    auto chart = v->Get("Chart");
    //    auto patch = v->Get("Patch");
    //    auto dx = chart->GetValue<nTuple<Real, 3>>("Scale");
    //    auto x0 = chart->GetValue<nTuple<Real, 3>>("Origin");
    //    if (chart == nullptr || patch == nullptr) { return 0; }
    //    if (auto atlas = v->Get("Atlas")) { auto domain = XDMFFromAtlas(v); }

    if (url.empty()) { url = v->GetValue<std::string>("Name", "simpla_unnamed") + ".xmf"; }
    auto writer = XdmfWriter::New(url);
    domain->accept(writer);

    return SP_SUCCESS;
}

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
//// int DataBaseXDMF::Foreach(std::function<int(std::string const&, std::shared_ptr<DataEntity>)> const& fun) const
///{
////    return 0;
////}

}  // namespace data{
}  // namespace simpla