//
// Created by salmon on 17-9-14.
//

#include <XdmfDomain.hpp>
#include <XdmfGrid.hpp>
#include <XdmfGridCollection.hpp>
#include <XdmfWriter.hpp>
#include <iostream>

int main(int argc, char** argv) {
    auto domain = XdmfDomain::New();
    auto grid_collection = XdmfGridCollection::New();
    auto grid_curv = XdmfCurvilinearGrid::New(4, 2, 2);
    grid_curv->setName("Curv");
    auto geo = XdmfGeometry::New();
    geo->setType(XdmfGeometryType::XYZ());
    geo->setOrigin(0, 0, 0);
    double x[4 * 2 * 2];
    double y[4 * 2 * 2];
    double z[4 * 2 * 2];
    for (int i = 0; i < 4 * 2 * 2; ++i) {
        x[i] = i % 4;
        y[i] = i % 2;
        z[i] = i % 2;
    }

    std::vector<unsigned int> dimensions = {4, 2, 2, 3};
    geo->initialize(XdmfArrayType::Float64(), dimensions);
    geo->insert(0, x, 4 * 2 * 2, 3, 1);
    geo->insert(1, y, 4 * 2 * 2, 3, 1);
    geo->insert(2, z, 4 * 2 * 2, 3, 1);

    auto dims = geo->getDimensions();
    std::cout << dims.size() << " [" << dims[0] << "," << dims[1] << "]" << std::endl;
    grid_curv->setGeometry(geo);
    grid_collection->insert(grid_curv);
    domain->insert(grid_collection);

    auto writer = XdmfWriter::New("test.xdmf");
    domain->accept(writer);
}