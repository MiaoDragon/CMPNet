/**
* Adapted from OMPL tutorial
**/
#include "ompl/geometric/planners/rrt/RRT.h"
#include <limits>
#include "ompl/base/goals/GoalSampleableRegion.h"
#include "ompl/tools/config/SelfConfig.h"
#include "ompl/base/Goal.h"
#include "ompl/base/goals/GoalSampleableRegion.h"
#include "ompl/base/goals/GoalState.h"
#include "ompl/base/objectives/PathLengthOptimizationObjective.h"
#include "ompl/base/samplers/InformedStateSampler.h"
#include "ompl/base/samplers/informed/RejectionInfSampler.h"
#include "ompl/base/samplers/informed/OrderedInfSampler.h"
#include "ompl/tools/config/SelfConfig.h"
#include "ompl/util/GeometricEquations.h"
#include "ompl/base/spaces/RealVectorStateSpace.h"
#include <ompl/base/goals/GoalStates.h>
#include <omplapp/apps/SE3RigidBodyPlanning.h>
#include <omplapp/config.h>
#include "mpnet_planner.hpp"
#include <ompl/base/spaces/SE3StateSpace.h>

#include <torch/torch.h>
#include <torch/script.h>
#include "mpnet_planner.hpp"
#include <iostream>
#include <cmath>
#include <iterator>
#include <fstream>
using namespace ompl;

int main()
{


    // debug if model output the same
    std::shared_ptr<torch::jit::script::Module> encoder(new torch::jit::script::Module(torch::jit::load("../encoder_annotated_test_cpu_2.pt")));
    // obtain obstacle representation
    std::vector<torch::jit::IValue> inputs;
    // variable for loading file
    std::ifstream infile;
    std::string pcd_fname = "../obs_voxel.txt";
    std::cout << "PCD file: " << pcd_fname << "\n\n\n";
    infile.open(pcd_fname);

    std::string line;
    std::vector<float> tt;
    while (getline(infile, line)){
        tt.push_back(std::atof(line.c_str()));
    }
    torch::Tensor torch_tensor = torch::from_blob(tt.data(), {1,1,32,32,32});
    #ifdef DEBUG
        std::cout << "after reading in obs and store in torch tensor" << std::endl;
    #endif
    inputs.push_back(torch_tensor);
    at::Tensor obs_enc = encoder->forward(inputs).toTensor();
    //torch::Tensor res = mlp_output.toTensor().to(at::kCPU);
    auto res_enc = obs_enc.accessor<float,2>(); // accesor for the tensor
    std::vector<float> encoder_out;
    for (int i=0; i < 64; i++)
    {
        encoder_out.push_back(res_enc[0][i]);
    }
    infile.close();
    std::ofstream outfile_test;
    outfile_test.open("obs_enc_cpp.txt");
    // write the mlpout to file
    for (int i=0; i < 64; i++)
    {
        outfile_test << encoder_out[i] << "\n";
    }
    //std::ostream_iterator<std::string> encoder_iter(outfile, " ");
    //std::copy(encoder_out.begin(), encoder_out.end(), encoder_iter);
    outfile_test.close();





    std::shared_ptr<torch::jit::script::Module> MLP(new torch::jit::script::Module(torch::jit::load("../mlp_annotated_test_gpu_2.pt")));
    MLP->to(at::kCUDA);
    infile.open("test_sample.txt");
    std::string input;
    tt.clear();
    while (getline(infile, input)){
        tt.push_back(std::atof(input.c_str()));
    }
    torch::Tensor mlp_input_tensor = torch::from_blob(tt.data(), {1,78});
    std::vector<torch::jit::IValue> mlp_input;
    mlp_input.push_back(mlp_input_tensor);

    outfile_test.open("test_sample_output_cpp.txt");
    std::vector<float> mlp_out;

    for (int i=0; i < 10; i++)
    {
        // generate 10 outputs
        auto mlp_output = MLP->forward(mlp_input);
        torch::Tensor res = mlp_output.toTensor().to(at::kCPU);
        auto res_a = res.accessor<float,2>(); // accesor for the tensor
        for (int j=0; i < 7; j++)
        {
            mlp_out.push_back(res_a[0][j]);
        }
    }
    // write the mlpout to file
    for (int i=0; i < 70; i++)
    {
        outfile_test << mlp_out[i] << "\n";
    }

    //std::ostream_iterator<std::string> mlp_iter(outfile, " ");
    //std::copy(mlp_out.begin(), mlp_out.end(), mlp_iter);
    outfile_test.close();



    // plan in SE3
    app::SE3RigidBodyPlanning setup;

    // load the robot and the environment
    std::string robot_fname = std::string(OMPLAPP_RESOURCE_DIR) + "/3D/Home_robot.dae";
    std::string env_fname = std::string(OMPLAPP_RESOURCE_DIR) + "/3D/Home_env.dae";
    setup.setRobotMesh(robot_fname);
    setup.setEnvironmentMesh(env_fname);

    // define start state
    base::ScopedState<base::SE3StateSpace> start(setup.getSpaceInformation());
    start->setX(262.95);
    start->setY(75.05);
    start->setZ(46.19);
    start->rotation().setIdentity();

    // define goal state
    base::ScopedState<base::SE3StateSpace> goal(start);
    goal->setX(200.49);
    goal->setY(-40.62);
    goal->setZ(46.19);
    goal->rotation().setIdentity();

    // set the start & goal states
    setup.setStartAndGoalStates(start, goal);

    // setting collision checking resolution to 1% of the space extent
    setup.getSpaceInformation()->setStateValidityCheckingResolution(0.01);

    // planner
    MPNetPlanner* planner = new MPNetPlanner(setup.getSpaceInformation(), false, 1001, 3000);
    base::PlannerPtr planner_ptr(planner);
    setup.setPlanner(planner_ptr);
    // we call setup just so print() can show more information
    setup.setup();
    setup.print();

    // try to solve the problem
    std::filebuf fb;
    fb.open("planned_path.txt", std::ios::out);
    std::ostream outfile(&fb);
    //std::string path_fname = "planned_path.txt";
    //outfile.open(path_fname);
    //if (setup.solve(10))
    //{
    //    // simplify & print the solution
    //    setup.simplifySolution();
    //    setup.getSolutionPath().printAsMatrix(infile);
    //}
    setup.solve(120);
    setup.getSolutionPath().printAsMatrix(outfile);
    fb.close();
    return 0;
}
