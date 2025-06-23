#include <iostream>

void fun(){
    static int count = 0;
    std::cout << "Count: " << count << std::endl;
    count++;
}

int main(){
    for(int i=0; i<10; i++){
        fun();
    }
    return 0;
}