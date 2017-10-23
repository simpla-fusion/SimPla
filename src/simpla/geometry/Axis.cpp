//
// Created by salmon on 17-10-23.
//
#include "Axis.h"
namespace simpla {
namespace geometry {

void Axis::Deserialize(std::shared_ptr<simpla::data::DataNode> const &cfg) {
    o = cfg->GetValue("Origin", o);
    x = cfg->GetValue("XAxis", x);
    y = cfg->GetValue("YAxis", y);
    z = cfg->GetValue("ZAxis", z);
}
std::shared_ptr<simpla::data::DataNode> Axis::Serialize() const {
    auto res = simpla::data::DataNode::New(simpla::data::DataNode::DN_TABLE);
    res->SetValue("Origin", o);
    res->SetValue("XAxis", x);
    res->SetValue("YAxis", y);
    res->SetValue("ZAxis", z);
    return res;
}

}  // namespace geometry{
}  // namespace simpla{