#include <iostream>
#include "example.pb.h"

void createSchool() {
//  auto school = std::make_shared<School>();
//  Class* sl = school->mutable_class_s();
  auto sl = std::make_shared<School>()->mutable_class_s();
  sl->set_id(1);

  for (int i = 1; i <= 2; ++i) {
    Student* st = sl->add_student();
    st->set_id(i);
    st->set_name("");
  }

  std::cout << "class id: " << sl->id() << std::endl;
  for (int i = 0; i < sl->student_size(); ++i) {
    auto st = sl->student(i);
    std::cout << "id: " << st.id() << " , name: " << st.name() << std::endl;
  }

}

int main() {
  GOOGLE_PROTOBUF_VERIFY_VERSION;
  createSchool();
  google::protobuf::ShutdownProtobufLibrary();
  return 0;
}
