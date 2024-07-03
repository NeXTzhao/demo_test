//
// Created by next on 2024/7/2.
//
#include <iostream>
#include <string>
#include <vector>

int main() {
  std::vector<std::string> name{"a", "b", "c"};

  for (int i = 0; i < 10; ++i) {
   auto s = name[i]; // 这样会报std::logic_error的错误
   if(s.empty()){
    // if (!name[i].empty()) {
      std::cout << name[i] << std::endl;
    }
  }
}