#include <iostream>
#include <vector>

class MyObject {
 public:
  MyObject(int value) : value_(value) {
    data_ = new int[100];  // 动态分配内存
  }
  ~MyObject() {
    // 析构函数中应释放动态分配的内存
    delete[] data_;
  }
  int getValue() const { return value_; }

 private:
  int value_;
  int* data_;
};

void createObjects(int num) {
  std::vector<MyObject*> objects;
  for (int i = 0; i < num; ++i) {
    objects.push_back(new MyObject(i));
  }
  // 在这里有意不释放 objects 中的对象，模拟内存泄漏
//  for (auto obj : objects) {
//    delete obj;
//  }
//  objects.clear();
}

int main() {
  createObjects(100);
  std::cout << "Program finished." << std::endl;
  return 0;
}
