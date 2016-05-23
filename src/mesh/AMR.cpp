//
// Created by salmon on 16-5-23.
//

void simpla::mesh::AMR::apply(uuid id, MeshWorker const &worker, Real dt)
{
    w.view(id);

    int ratio = m_mesh_atlas_.refine_ratio(id);

    auto children = m_mesh_atlas_.children(id);

    Real sub_dt = dt / ratio;

    // copy data to lower level
    for (auto &sub_id:children) { w.refine(sub_id); }

    // push lower level
    for (int n = 0; n < ratio; ++n)
    {
        for (auto &sub_id:children) { apply(w, sub_id, sub_dt); }

        for (auto &sub_id:children)
        {
            // move to lower level
            w.view(sub_id);

            for (auto const &oid:m_mesh_atlas_.sibling(sub_id))
            {
                w.sync(oid);
            }
        }

        // TODO: add mpi sync at here
    }
    w.view(id);
    //copy data from lower level
    for (auto &sub_id:children) { w.coarsen(sub_id); }
    // push this level
    w.work(dt);
}