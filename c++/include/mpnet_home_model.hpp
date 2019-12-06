#ifndef MPNET_HOME_MODEL_
#define MPNET_HOME_MODEL_
#include <torch/torch.h>

using namespace torch;
// Define a new Module.
struct MLP : nn::Module {
  MLP(input_size, output_size)
  {
    // Construct and register two Linear submodules.
    fc = nn::Sequential(
        // Layer 1
        nn::Linear(input_size, 2560), nn::PReLU(), nn::Dropout(),
        nn::Linear(2560, 1792), nn::PReLU(), nn::Dropout(),
        nn::Linear(1792, 1024), nn::PReLU(), nn::Dropout(),
        nn::Linear(1024, 512), nn::PReLU(), nn::Dropout(),
        nn::Linear(512, 256), nn::PReLU(), nn::Dropout(),
        nn::Linear(256, 128), nn::PReLU(), nn::Dropout(),
        nn::Linear(128, 64), nn::PReLU(),
        nn::Linear(64, output_size));
  }

  // Implement the Net's algorithm.
  torch::Tensor forward(torch::Tensor x)
  {
    return fc->forward(x);
  }
  // Use one of many "standard library" modules.
  nn::Sequential fc{nullptr};
};

struct Encoder : nn::Module {
    Encoder(input_size, output_size)
    {
      // Construct and register two Linear submodules.
      encoder = nn::Sequential(
          // Layer 1
          nn::Conv3d(nn::Conv3dOptions(1,16,6).stride(1).padding(0).with_bias(true)),
          nn::PReLU(),
          nn::MaxPool3d(nn::MaxPool3dOptions(2).stride(2)),
          nn::Conv3d(nn::Conv3dOptions(16,8,3).stride(2),padding(0).with_bias(true)),
          nn::PReLU()
      );
      // obtain the size before fc
      torch::Tensor x = encoder(torch::rand({1,1,input_size,input_size,input_size}));
      int first_fc_in_features = 1;
      std::vector<int> size = x.sizes();
      for (int i = 1; i < size.size(); i++)
      {
          first_fc_in_features *= size[i];
      }
      head = nn.Sequential(
          nn::Linear(first_fc_in_features, otput_size)
      );
    }
    // Implement the Net's algorithm.
    torch::Tensor forward(torch::Tensor x)
    {
      x = encoder->forward(x);
      x = x.view({x.sizes()[0], -1});
      x = head->forward(x);
      return x
    }
    // Use one of many "standard library" modules.
    nn::Sequential encoder{nullptr}, head{nullptr};
};

struct MPNet: nn::Module
{
    MPNet(encoder_input_size, encoder_output_size, mlp_input_size, output_size)
    {
        mlp = MLP(mlp_input_size, output_size);
        encoder = Encoder(encoder_input_size, encoder_output_size);
    }
    torch::Tensor forward(torch::Tensor x, torch::Tensor obs)
    {
        torch::Tensor z = encoder(obs);
        torch::Tensor mlp_in = torch::cat({z,x}, 1);
        return mlp(mlp_in);
    }
    nn::Module mlp{nullptr}, encoder{nullptr};
}

#endif
