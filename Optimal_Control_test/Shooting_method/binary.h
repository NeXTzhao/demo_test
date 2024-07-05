#ifndef CYBER_BINARY_H_
#define CYBER_BINARY_H_

#include <string>

namespace cyber {
class Binary {
public:
    static std::string GetName() { return GetNameRef(); }
    static void SetName(const std::string &name) { GetNameRef() = name; }
    static std::string &GetNameRef() {
        static std::string binary_name;
        return binary_name;
    }
};
}  // namespace cyber

#endif  // CYBER_BINARY_H_
