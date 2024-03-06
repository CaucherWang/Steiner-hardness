#include<iostream>
#include <fstream>
#include<vector>

class NSG{
public:
    int width, ep_, nd_;
    std::vector<std::vector<int > > final_graph_;

    void Load(const char *filename);
};


void NSG::Load(const char *filename) {
  std::ifstream in(filename, std::ios::binary);
  if(!in.is_open()){
    std::cerr << "Error opening file " << filename << std::endl;
    exit(-1);
  }
  in.read((char *)&width, sizeof(unsigned));
  in.read((char *)&ep_, sizeof(unsigned));
  // width=100;
  unsigned cc = 0;
  while (!in.eof()) {
    unsigned k;
    in.read((char *)&k, sizeof(unsigned));
    if (in.eof()) break;
    cc += k;
    std::vector<unsigned> tmp(k);
    in.read((char *)tmp.data(), k * sizeof(unsigned));
    std::vector<int> tmp2(k);
    for (int i = 0; i < k; i++) {
      tmp2[i] = tmp[i];
    }
    final_graph_.push_back(tmp2);
  }
  nd_ = final_graph_.size();
  std::cerr << " ep = " << ep_ << " nd = " << final_graph_.size() << std::endl;
  std::cerr << "NSG edges number = " << cc << std::endl;
  in.close();
  // std::cout<<cc<<std::endl;
}