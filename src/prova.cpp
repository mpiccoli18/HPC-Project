#include <iostream>

using namespace std;
int main(){
    int i;
    int j = 100;
    for(i = 0; i < 10; i++)
    {
        cout << "This is just an output of " + i;
        cout << endl << "Also, there is " + j << "endl";
    }
    return 0;
}