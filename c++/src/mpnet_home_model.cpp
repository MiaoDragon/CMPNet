#include "mpnet_home_model.hpp"

// test code for loading the model from file
int main()
{
    MPNet mpnet(32, 64, 78, 7);
    // load trained model
    // confirmed that this won't work in C++ as Python needs Pickle
    torch::load()
}
