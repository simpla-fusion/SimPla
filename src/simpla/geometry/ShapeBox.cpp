//
// Created by salmon on 17-11-14.
//
#include "ShapeBox.h"
namespace simpla {
namespace geometry {
void ShapeBoxBase::Deserialize(std::shared_ptr<simpla::data::DataNode> const &cfg) {
    std::get<0>(m_parameter_box_) = cfg->GetValue("Min", std::get<0>(m_parameter_box_));
    std::get<1>(m_parameter_box_) = cfg->GetValue("Max", std::get<1>(m_parameter_box_));
}
std::shared_ptr<simpla::data::DataNode> ShapeBoxBase::Serialize() const {
    auto res = data::DataNode::New(data::DataNode::DN_TABLE);
    res->SetValue("Min", std::get<0>(m_parameter_box_));
    res->SetValue("Max", std::get<0>(m_parameter_box_));
    return res;
}
}  // namespace geometry
}  // namespace simpla