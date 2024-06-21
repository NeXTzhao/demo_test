#include <iostream>
#include "example.pb.h"

void createSchool(const std::shared_ptr<School>&school) {
//  auto school = std::make_shared<School>();
//  Class* sl = school->mutable_class_s();
  auto sl = school->mutable_class_s();
  sl->set_id(1);

  for (int i = 1; i <= 2; ++i) {
    Student* st = sl->add_student();
    st->set_id(i);

    /*
     * 报错：
     * terminate called after throwing an instance of 'std::logic_error'
        what():  basic_string::_M_construct null not valid
     * */
//    char* a = nullptr;
    std::string ss ;
    ss += "11";

//    st->set_name(ss);
  }

  std::cout << "class id: " << sl->id() << std::endl;
  for (int i = 0; i < sl->student_size(); ++i) {
    auto st = sl->student(i);
    std::cout << "id: " << st.id() << " , name: " << st.name() << std::endl;
  }

}

int main() {
  GOOGLE_PROTOBUF_VERIFY_VERSION;
  auto sl = std::make_shared<School>();
  createSchool(sl);
  google::protobuf::ShutdownProtobufLibrary();
  return 0;
}
