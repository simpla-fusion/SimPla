//
// Created by salmon on 16-11-27.
//
#include "Model.h"

namespace simpla { namespace model
{
void Model::load(std::string const &)
{
    // TODO: load geometry object to m_geo_obj_;
    UNIMPLEMENTED;
};

void Model::save(std::string const &)
{
    UNIMPLEMENTED;
};

void Model::deploy()
{
    auto m_coord_ = chart->coordinate_frame();
//    m_tags_.move_to(m_coord_->mesh_block());
//    m_fraction_.move_to(m_coord_->mesh_block());
//    m_dual_fraction_.move_to(m_coord_->mesh_block());
//    m_tags_.deploy();
//    m_fraction_.deploy();
//    m_dual_fraction_.deploy();

};

void Model::initialize(Real data_time)
{
    deploy();

//    m_tags_.clear();
//    m_fraction_.clear();
//    m_dual_fraction_.clear();
//
//    auto m_coord_ = chart->coordinate_frame();
//
////    auto r = m_coord_->range(VERTEX, m_geo_->box());
////
////    if (r.size() > 0)
////    {
////        r.foreach(
////                [&](MeshEntityId const &s)
////                {
////                    if (m_geo_->within(m_coord_->point(s))) { tag(s) = 1; }
////                }
////        );
////    }
//    auto b = toolbox::intersection(m_coord_->box(), m_geo_->box());

};
}}