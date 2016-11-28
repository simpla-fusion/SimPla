//
// Created by salmon on 16-11-27.
//
#include <simpla/SIMPLA_config.h>
#include <simpla/toolbox/nTuple.h>
#include <simpla/model/GEqdsk.h>


namespace simpla
{
void convert_geqdsk2step(GEqdsk const &geqdsk, std::string const &filename);
}
using namespace simpla;

int main(int argc, char **argv)
{
    GEqdsk gEqdsk;
    std::string input_file = "geqdsk.gfile";
    std::string output_file = "geqdsk";
    if (argc == 0)
    {
        std::cout << " Usage: " << argv[0] << " <input file> <output file> " << std::endl;
        exit(1);
    } else if (argc >= 2) { input_file = argv[1]; }
    else if (argc >= 3) { output_file = argv[2]; }

    std::cout << " input  :" << input_file << std::endl;

    gEqdsk.load(input_file);
    std::cout << " output :" << output_file << std::endl;

    convert_geqdsk2step(gEqdsk, output_file);
}