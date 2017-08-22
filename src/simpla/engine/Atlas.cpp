/**
 *  @file MeshAtlas.cpp
 *  Created by salmon on 16-5-23.
 */

#include "simpla/SIMPLA_config.h"

#include "simpla/geometry/Chart.h"

#include "Atlas.h"
#include "Patch.h"

//#include "TransitionMap.h"
//#include "simpla/utilities/BoxUtility.h"

namespace simpla {
namespace engine {

struct Atlas::pimpl_s {
    std::map<id_type, std::shared_ptr<Patch>> m_patches_;

    static constexpr int MAX_NUM_OF_LEVEL = 5;

    typedef typename std::multimap<id_type, id_type>::iterator link_iterator;
    typedef typename std::multimap<id_type, id_type>::const_iterator const_link_iterator;
    typedef std::pair<const_link_iterator, const_link_iterator> multi_links_type;
    std::multimap<id_type, id_type> m_adjacent_;
    std::multimap<id_type, id_type> m_refine_;
    std::multimap<id_type, id_type> m_coarsen_;

    int m_level_ = 0;
    int m_max_level_ = 2;

    std::set<std::shared_ptr<Patch>> m_layers_[MAX_NUM_OF_LEVEL];

    nTuple<int, 3> m_refine_ratio_[MAX_NUM_OF_LEVEL] = {{2, 2, 2}, {2, 2, 2}, {2, 2, 2}, {2, 2, 2}, {2, 2, 2}};
    nTuple<int, 3> m_smallest_dimensions_{8, 8, 8};
    nTuple<int, 3> m_largest_dimensions_{64, 64, 64};
    nTuple<int, 3> m_periodic_dimensions_{1, 1, 1};
    nTuple<int, 3> m_coarsest_dimensions_{1, 1, 1};
    index_box_type m_coarsest_index_box_{{0, 0, 0}, {1, 1, 1}};
};

Atlas::Atlas() : m_pimpl_(new pimpl_s) { SPObject::SetName("Atlas"); };
Atlas::~Atlas() { delete m_pimpl_; }

void Atlas::Serialize(std::shared_ptr<data::DataNode> const &cfg) const {
    base_type::Serialize(cfg);
    auto tdb = std::dynamic_pointer_cast<data::DataTable>(cfg);
    if (tdb != nullptr) {
        tdb->SetValue("PeriodicDimension", GetPeriodicDimensions());
        tdb->SetValue("CoarsestIndexBox", GetCoarsestIndexBox());

        tdb->SetValue("MaxLevel", GetMaxLevel());
        tdb->SetValue("RefineRatio", GetRefineRatio(0));
        tdb->SetValue("LargestPatchDimensions", GetLargestPatchDimensions());
        tdb->SetValue("SmallestPatchDimensions", GetSmallestPatchDimensions());
    }
};
void Atlas::Deserialize(std::shared_ptr<const data::DataNode> const &cfg) {
    base_type::Deserialize(cfg);
    auto tdb = std::dynamic_pointer_cast<const data::DataTable>(cfg);
    if (tdb != nullptr) {
        m_pimpl_->m_periodic_dimensions_ = tdb->GetValue<nTuple<int, 3>>("PeriodicDimension", nTuple<int, 3>{0, 0, 0});
        //    std::get<0>(m_pimpl_->m_coarsest_index_box_) =tdb->Get("CoarsestIndexBox/lo", nTuple<int, 3>{0, 0,
        //    0});
        std::get<1>(m_pimpl_->m_coarsest_index_box_) = tdb->GetValue("Dimensions", nTuple<int, 3>{1, 1, 1});
        m_pimpl_->m_max_level_ = tdb->GetValue<int>("MaxLevel", 1);

        m_pimpl_->m_smallest_dimensions_ =
            tdb->GetValue<nTuple<int, 3>>("SmallestPatchDimensions", nTuple<int, 3>{4, 4, 4});

        m_pimpl_->m_largest_dimensions_ =
            tdb->GetValue<nTuple<int, 3>>("LargestPatchDimensions", nTuple<int, 3>{128, 128, 128});

        m_pimpl_->m_refine_ratio_[0] = tdb->GetValue<nTuple<int, 3>>("RefineRatio", nTuple<int, 3>{2, 2, 2});
    }
    Click();
};

size_type Atlas::DeletePatch(id_type id) { return m_pimpl_->m_patches_.erase(id); }

id_type Atlas::SetPatch(const std::shared_ptr<Patch> &p) {
    auto res = m_pimpl_->m_patches_.emplace(p->GetMeshBlock()->GetGUID(), p);
    if (!res.second) { res.first->second = p; }
    return res.first->first;
}
std::shared_ptr<Patch> Atlas::GetPatch(id_type id) {
    std::shared_ptr<Patch> res = nullptr;
    auto it = m_pimpl_->m_patches_.find(id);
    if (it != m_pimpl_->m_patches_.end()) { res = it->second; }
    return res;
}
std::shared_ptr<Patch> Atlas::GetPatch(const std::shared_ptr<MeshBlock> &mblk) {
    auto res = GetPatch(mblk->GetGUID());
    if (res == nullptr) { res = GetPatch(SetPatch(Patch::New(mblk))); }
    // else { TODO: check mblk.index_box}
    return res;
};

std::shared_ptr<const Patch> Atlas::GetPatch(id_type id) const {
    std::shared_ptr<const Patch> res = nullptr;
    auto it = m_pimpl_->m_patches_.find(id);
    if (it != m_pimpl_->m_patches_.end()) { res = it->second; }
    return res;
}

int Atlas::GetNumOfLevel() const { return m_pimpl_->m_max_level_; }
int Atlas::GetMaxLevel() const { return m_pimpl_->m_max_level_; }
void Atlas::SetMaxLevel(int l) {
    m_pimpl_->m_max_level_ = l;
    Click();
}
void Atlas::SetRefineRatio(nTuple<int, 3> const &v, int level) {
    m_pimpl_->m_refine_ratio_[level] = v;
    m_pimpl_->m_level_ = std::max(level, m_pimpl_->m_level_);
    Click();
}
nTuple<int, 3> Atlas::GetRefineRatio(int l) const { return m_pimpl_->m_refine_ratio_[l]; };
void Atlas::SetLargestPatchDimensions(nTuple<int, 3> const &d) { m_pimpl_->m_largest_dimensions_ = d; };
nTuple<int, 3> Atlas::GetLargestPatchDimensions() const { return m_pimpl_->m_largest_dimensions_; };
void Atlas::SetSmallestPatchDimensions(nTuple<int, 3> const &d) { m_pimpl_->m_smallest_dimensions_ = d; };
nTuple<int, 3> Atlas::GetSmallestPatchDimensions() const { return m_pimpl_->m_smallest_dimensions_; };

void Atlas::SetPeriodicDimensions(nTuple<int, 3> const &b) { m_pimpl_->m_periodic_dimensions_ = b; }
nTuple<int, 3> const &Atlas::GetPeriodicDimensions() const { return m_pimpl_->m_periodic_dimensions_; }

void Atlas::SetCoarsestIndexBox(index_box_type const &b) { m_pimpl_->m_coarsest_index_box_ = b; }
index_box_type const &Atlas::GetCoarsestIndexBox() const { return m_pimpl_->m_coarsest_index_box_; }

//        size_tuple Atlas::GetDimensions() const {
//            size_tuple d;
//            d = std::get<1>(m_pimpl_->m_index_box_) - std::get<0>(m_pimpl_->m_index_box_);
//            return d;
//        }
// size_type Atlas::size(int level) const { return m_database_->m_layer_[level].size(); }
// void Atlas::max_level(int ml) { m_database_->m_max_level_ = ml; }
// int Atlas::max_level() const { return m_database_->m_max_level_; }
// bool Atlas::has(id_type id) const { return m_database_->m_nodes_.find(id) != m_database_->m_nodes_.end(); };
//
// RectMesh *Atlas::find(id_type id) {
//    auto it = m_database_->m_nodes_.find(id);
//    if (it != m_database_->m_nodes_.end()) {
//        return it->second.get();
//    } else {
//        return nullptr;
//    }
//}
//
// RectMesh const *Atlas::find(id_type id) const {
//    auto it = m_database_->m_nodes_.find(id);
//    if (it != m_database_->m_nodes_.end()) {
//        return it->second.get();
//    } else {
//        return nullptr;
//    }
//}
//
// RectMesh *Atlas::at(id_type id) { return (m_database_->m_nodes_.at(id)).get(); };
//
// RectMesh const *Atlas::at(id_type id) const { return (m_database_->m_nodes_.at(id)).get(); };
//
// RectMesh const *Atlas::Connect(std::shared_ptr<RectMesh> const &p_m, RectMesh const *hint) {
////    m_pimpl_->m_nodes_.emplace(std::make_pair(p_m->id(), p_m));
//    return p_m.get();
//};
//
// std::ostream &Atlas::Print(std::ostream &os, int indent) const {
////    os << std::setw(indent) << "*" << name() << std::endl;
////    for (auto const &item : m_pimpl_->m_nodes_) {
////        os << "|" << std::setw(indent + 5 + item.second->level()) << std::setfill('-') << "> " << std::setfill(' ')
////           << std::setw(10) << std::left << item.first << std::right << " = {";
////        item.second->Print(os, indent + 1);
////        os << "}," << std::endl;
////    }
//    return os;
//};
//
// void Atlas::Load(const data::DataTable &) { UNIMPLEMENTED; }
//
// void Atlas::Save(data::DataTable *) const { UNIMPLEMENTED; }
//
// void Atlas::Sync(id_type id)
//{
//    unlink(id);
//
//    auto it = m_nodes_.find(id);
//
//    if (it == m_nodes_.end())
//    {
//        return;
//    } else if (id != it->second->id())
//    {
//        m_nodes_.Finalizie(it);
//        return;
//    }
//    RectMesh const &m = *(it->second);
//    assert(m.level() < MAX_NUM_OF_LEVEL);
//    m_max_level_ = std::max(m_max_level_, m.level());
//    m_layer_[m.level()].Connect(id);
//
//    for (auto const &item:m_nodes_) { if (item.first != id) { link(id, item.first); }}
//}
//
// void Atlas::link(id_type i0, id_type i1)
//{
//    assert(has(i0) && find(i1));
//    RectMesh const &m0 = *at(i0);
//    RectMesh const &m1 = *at(i1);
//    int l0 = m0.level();
//    int l1 = m1.level();
//    box_type b0 = m0.box();
//    box_type b1 = m1.box();
//    vector_type dx = m0.dx();
//    vector_type L;//= m0.period_length();
//    switch (l0 - l1)
//    {
//        case 0:
//            if (utilities::check_adjoining(b0, b1, dx, L))
//            {
//                m_adjacent_.Connect(i0, i1);
//                m_adjacent_.Connect(i1, i0);
//            }
//            break;
//
//        case -1:
//            if (utilities::check_overlapping(b0, b1))
//            {
//                m_refine_.Connect(i0, i1);
//                m_coarsen_.Connect(i1, i0);
//            }
//            break;
//        case 1:
//            if (utilities::check_overlapping(b0, b1))
//            {
//                m_coarsen_.Connect(i0, i1);
//                m_refine_.Connect(i1, i0);
//            }
//            break;
//        default:
//            break;
//    }
//
//    return l0 - l1;
//}
//
// void Atlas::unlink(id_type id)
//{
//    m_adjacent_.Finalizie(id);
//    m_refine_.Finalizie(id);
//    m_coarsen_.Finalizie(id);
//    for (int i = 0; i < MAX_NUM_OF_LEVEL; ++i) { m_layer_[i].Finalizie(id); }
//}
//
//
// void Atlas::update_all()
//{
//    for (auto ib = m_nodes_.begin(), ie = m_nodes_.end(); ib != ie; ++ib)
//        for (auto it = ib; it != ie; ++it) { link(ib->first, it->first); }
//    m_max_level_ = 0;
//    for (unsigned int i = 0; i < MAX_NUM_OF_LEVEL; ++i)
//    {
//        if (m_layer_[i].size() == 0) { break; }
//        m_max_level_ = i;
//    }
//
//}
//
//
// std::shared_ptr<TransitionMapBase>
// Atlas::add_adjacency(MeshBlockId first, MeshBlockId second)
//{
//    return add_adjacency(get_block(first), get_block(second));
//}
//
// std::shared_ptr<TransitionMapBase>
// Atlas::add_adjacency(std::shared_ptr<const RectMesh> first, std::shared_ptr<const RectMesh> second)
//{
//    return add_adjacency(std::make_shared<TransitionMapBase>(*first, *second));
//}
//
// std::tuple<std::shared_ptr<TransitionMapBase>, std::shared_ptr<TransitionMapBase>>
// Atlas::add_connection(std::shared_ptr<const RectMesh> first, std::shared_ptr<const RectMesh> second)
//{
//    return std::make_tuple(add_adjacency(first, second), add_adjacency(second, first));
//}
}  // namespace engine
}  // namespace simpla{namespace mesh_as{