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
template <typename U, int N>
std::shared_ptr<XdmfArray> XDMFNewArray(nTuple<U, N> const& v){};

boost::shared_ptr<XdmfCurvilinearGrid> XDMFCurvilinearGridNew(std::shared_ptr<DataNode> const& chart,
                                                              std::shared_ptr<data::DataNode> const& patch) {
    auto grid = XdmfCurvilinearGrid::New(4, 2, 2);
    auto geo = XdmfGeometry::New();
    geo->setType(XdmfGeometryType::XYZ());
    auto origin = chart->GetValue<nTuple<Real, 3>>("Origin");
    geo->setOrigin(origin[0], origin[1], origin[2]);
    grid->setGeometry(geo);

    //    if (auto attrs = patch->Get("Attributes")) {
    //        attrs->Foreach([&](std::string const& k, std::shared_ptr<data::DataNode> const& node) { return 1; });
    //    }
    return grid;
}
boost::shared_ptr<XdmfRegularGrid> XDMFRegularGridNew(std::shared_ptr<DataNode> const& chart,
                                                      std::shared_ptr<data::DataNode> const& patch) {
    auto x0 = chart->GetValue<nTuple<Real, 3>>("Origin");
    auto dx = chart->GetValue<nTuple<Real, 3>>("Scale");
    auto idx_box = patch->GetValue<index_box_type>("Block");
    nTuple<unsigned int, 3> dims;
    dims = std::get<1>(idx_box) - std::get<0>(idx_box);
    nTuple<unsigned int, 3> origin;
    origin = std::get<0>(idx_box) * dx + x0;
    auto grid = XdmfRegularGrid::New(dx[0], dx[1], dx[2], dims[0], dims[1], dims[2], origin[0], origin[1], origin[2]);
    return grid;
}
int XDMFDump(std::string url, std::shared_ptr<DataNode> const& obj) {
    if (obj == nullptr || url.empty()) { return SP_FAILED; }
    int success = SP_FAILED;
    auto grid_collection = XdmfGridCollection::New();
    grid_collection->setType(XdmfGridCollectionType::Spatial());

    if (auto time = obj->Get("Time")) {
        if (auto t = std::dynamic_pointer_cast<DataLightT<Real>>(time->GetEntity())) {
            grid_collection->setTime(XdmfTime::New(t->value()));
        }
    }
    if (auto chart = obj->Get("Chart"))
        if (auto patch = obj->Get("Patch")) {
            patch->Foreach([&](std::string const& k, std::shared_ptr<data::DataNode> const& node) {
                if (node->Get("Coordinates") != nullptr) {
                    grid_collection->insert(XDMFCurvilinearGridNew(chart, node));
                } else {
                    grid_collection->insert(XDMFRegularGridNew(chart, node));
                }
                return 1;
            });
        }

    auto domain = XdmfDomain::New();
    domain->insert(grid_collection);
    if (auto writer = XdmfWriter::New(url)) {
        domain->accept(writer);
        success = SP_SUCCESS;
    }
    return success;
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