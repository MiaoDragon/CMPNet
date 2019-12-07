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
#include <sstream>
#include <cmath>
#include <iterator>
#include <fstream>
#include <stdio.h>


#include <chrono>
typedef std::chrono::high_resolution_clock Time;
typedef std::chrono::milliseconds ms;
typedef std::chrono::duration<float> fsec;
typedef std::vector<ompl::base::State *> StatePtrVec;



#define STATE_N 7


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
    outfile_test.open("../obs_enc_cpp.txt");
    // write the mlpout to file
    for (int i=0; i < 64; i++)
    {
        outfile_test << encoder_out[i] << "\n";
    }
    //std::ostream_iterator<std::string> encoder_iter(outfile, " ");
    //std::copy(encoder_out.begin(), encoder_out.end(), encoder_iter);
    outfile_test.close();


    std::cout << "finished encoder testing." << std::endl;


    std::shared_ptr<torch::jit::script::Module> MLP(new torch::jit::script::Module(torch::jit::load("../mlp_annotated_test_gpu_2.pt")));
    MLP->to(at::kCUDA);
    infile.open("../test_sample.txt");
    std::string input;
    tt.clear();
    while (getline(infile, input)){
        tt.push_back(std::atof(input.c_str()));
    }
    std::cout << "after loading data." << std::endl;
    torch::Tensor mlp_input_tensor = torch::from_blob(tt.data(), {1,78}).to(at::kCUDA);
    std::vector<torch::jit::IValue> mlp_input;
    mlp_input.push_back(mlp_input_tensor);

    outfile_test.open("../test_sample_output_cpp.txt");
    std::vector<float> mlp_out;
    std::cout << "before testing." << std::endl;
    for (int i=0; i < 100; i++)
    {
        // generate 10 outputs
        auto mlp_output = MLP->forward(mlp_input);
        torch::Tensor res = mlp_output.toTensor().to(at::kCPU);
        auto res_a = res.accessor<float,2>(); // accesor for the tensor
        for (int j=0; j < 7; j++)
        {
            mlp_out.push_back(res_a[0][j]);
        }
    }
    // write the mlpout to file
    for (int i=0; i < 700; i++)
    {
        outfile_test << mlp_out[i] << "\n";
    }

    //std::ostream_iterator<std::string> mlp_iter(outfile, " ");
    //std::copy(mlp_out.begin(), mlp_out.end(), mlp_iter);
    infile.close();
    outfile_test.close();

    std::cout << "finished mlp testing." << std::endl;

    //####################Finished testing################################
    //####################################################################

    // plan in SE3
    app::SE3RigidBodyPlanning setup;

    // load the robot and the environment
    std::string robot_fname = std::string(OMPLAPP_RESOURCE_DIR) + "/3D/Home_robot.dae";
    std::string env_fname = std::string(OMPLAPP_RESOURCE_DIR) + "/3D/Home_env.dae";
    setup.setRobotMesh(robot_fname);
    setup.setEnvironmentMesh(env_fname);

    MPNetPlanner* planner = new MPNetPlanner(setup.getSpaceInformation(), false, 1001, 3000);

    // setting collision checking resolution to 1% of the space extent
    setup.getSpaceInformation()->setStateValidityCheckingResolution(0.01);

    // planner
    //MPNetPlanner* planner = new MPNetPlanner(setup.getSpaceInformation(), false, 1001, 3000);
    base::PlannerPtr planner_ptr(planner);
    setup.setPlanner(planner_ptr);
    // we call setup just so print() can show more information
    setup.setup();
    //setup.print();



    const int sp = 2196;
    const int s = 0;
    const int N = 1;
    const int NP = 500;
    std::vector<float> plan_times;
    std::vector<float> plan_sucs;
    std::vector<float> plan_lens;
    std::vector<float> data_lens;

    std::string model_path = "/media/arclabdl1/HD1/YLmiao/results/CMPnet_res/home_mlp2_lr025_SGD_c++/";
    float accuracy = 0.;
    float num_suc = 0.;
    float num_total = 0.;
    for (int env_idx=0; env_idx<N; env_idx++)
    {
      int path_idx = 0;
      while (path_idx < NP)
      {
        // * load data
        //std::ifstream infile;
        auto load_t0 = Time::now();
        std::ifstream infile;
        std::string p_fname = "/media/arclabdl1/HD1/YLmiao/data/home/paths/path_" + std::to_string(path_idx+sp) + ".txt";
        path_idx += 1;
        infile.open(p_fname);
        //infile = fopen(p_fname, 'r');

        std::string line;
        std::vector<std::vector<float>> path;
        while (getline(infile, line)){
          if (line[0] == '\n' || (int)line[0] == 0)
          {
            continue;
          }
          std::vector<float> state;
          std::stringstream ss(line);
          for (int state_i=0; state_i < STATE_N; state_i ++)
          {
            float data;
            ss >> data;
            state.push_back(data);
          }
          path.push_back(state);
        }
        if (path.size() < 2)
        {
          continue;
        }
        auto load_t1 = Time::now();
        fsec time_load = load_t1 - load_t0;
        std::cout << "data loading time: " << time_load.count() << std::endl;
        // * setup env

        auto setup_t0 = Time::now();

        // define start state
        base::ScopedState<base::SE3StateSpace> start(setup.getSpaceInformation());

        std::vector<float> start_vec = path[0];
        std::vector<float> goal_vec = path.back();
        std::cout << "start: " << std::endl;
        std::cout << start_vec << std::endl;
        std::cout << "goal: " << std::endl;
        std::cout << goal_vec << std::endl;

        start->setX(start_vec[0]);
        start->setY(start_vec[1]);
        start->setZ(start_vec[2]);
        std::vector<float> angle;
        planner->q_to_axis_angle(start_vec[6], start_vec[3], start_vec[4], start_vec[5], angle);
        start->rotation().setAxisAngle(angle[0], angle[1], angle[2], angle[3]);
        // define goal state
        base::ScopedState<base::SE3StateSpace> goal(start);
        goal->setX(goal_vec[0]);
        goal->setY(goal_vec[1]);
        goal->setZ(goal_vec[2]);
        planner->q_to_axis_angle(goal_vec[6], goal_vec[3], goal_vec[4], goal_vec[5], angle);
        goal->rotation().setAxisAngle(angle[0], angle[1], angle[2], angle[3]);

        // set the start & goal states
        setup.setStartAndGoalStates(start, goal);
        //setup.setup();
        //setup.print();
        auto setup_t1 = Time::now();
        fsec time_setup = setup_t1 - setup_t0;
        std::cout << "set up env time: " << time_setup.count() << std::endl;



        // * plan
        // try to solve the problem

        auto plan_t0 = Time::now();
        base::PlannerStatus status = setup.solve(120);
        auto plan_t1 = Time::now();
        fsec time_plan = plan_t1 - plan_t0;
        float time_spent = time_plan.count();
        std::cout << "plan takes total time: " << time_spent << "s" << std::endl;



        float plan_suc = 0.0;
        if (status == base::PlannerStatus::EXACT_SOLUTION)
        {
          std::filebuf fb;
          fb.open(model_path+"paths/path_"+std::to_string(path_idx+sp)+"_fes.txt", std::ios::out);
          std::ostream outfile(&fb);
          setup.getSolutionPath().printAsMatrix(outfile);
          fb.close();
          plan_suc = 1.0;
        }
        else
        {
          std::filebuf fb;
          fb.open(model_path+"paths/path_"+std::to_string(path_idx+sp)+"_infes.txt", std::ios::out);
          std::ostream outfile(&fb);
          setup.getSolutionPath().printAsMatrix(outfile);
          fb.close();
          plan_suc = 0.0;
        }



        auto data_path_t0 = Time::now();
        // output the data path length and txt file

        auto data_path(std::make_shared<ompl::geometric::PathGeometric>(setup.getSpaceInformation()));
        for (int i=0; i<path.size(); i++)
        {
          base::State* state = setup.getSpaceInformation()->allocState();

          state->as<base::SE3StateSpace::StateType>()->setX(path[i][0]);
          state->as<base::SE3StateSpace::StateType>()->setY(path[i][1]);
          state->as<base::SE3StateSpace::StateType>()->setZ(path[i][2]);
          std::vector<float> angle;
          planner->q_to_axis_angle(path[i][6], path[i][3], path[i][4], path[i][5], angle);
          state->as<base::SE3StateSpace::StateType>()->rotation().setAxisAngle(angle[0], angle[1], angle[2], angle[3]);
          data_path->append(state);
          setup.getSpaceInformation()->freeState(state);
        }

        auto data_path_t1 = Time::now();
        fsec time_data_path = data_path_t1 - data_path_t0;
        std::cout << "data path constructing time: " << time_data_path.count() << std::endl;


        // obtain the evaluation for the path (accuracy, time, path length)
        plan_times.push_back(time_spent);
        plan_sucs.push_back(plan_suc);
        float path_len = setup.getSolutionPath().length();
        plan_lens.push_back(path_len);
        num_suc += plan_suc;
        num_total += 1.0;
        accuracy = num_suc / num_total;
        std::cout << "path length: " << path_len << std::endl;
        std::cout << "current accuracy: " << num_suc << "/" << num_total << " = " << accuracy << std::endl;

        data_lens.push_back(data_path->length());

      }
    }
    // write the evaluation metrics
    std::filebuf fb;
    fb.open(model_path+"plan_times.txt", std::ios::out);
    std::ostream time_f(&fb);
    //TODO: add env dimension in the vector
    for (int i=0; i < N*NP; i++)
    {
      time_f << plan_times[i] << "\n";
    }
    fb.close();

    fb.open(model_path+"plan_sucs.txt", std::ios::out);
    std::ostream suc_f(&fb);
    //TODO: add env dimension in the vector
    for (int i=0; i < N*NP; i++)
    {
      suc_f << plan_sucs[i] << "\n";
    }
    fb.close();

    fb.open(model_path+"plan_lens.txt", std::ios::out);
    std::ostream len_f(&fb);
    //TODO: add env dimension in the vector
    for (int i=0; i < N*NP; i++)
    {
      len_f << plan_lens[i] << "\n";
    }
    fb.close();


    fb.open(model_path+"data_lens.txt", std::ios::out);
    std::ostream data_len_f(&fb);
    //TODO: add env dimension in the vector
    for (int i=0; i < N*NP; i++)
    {
      data_len_f << data_lens[i] << "\n";
    }
    fb.close();



    fb.open(model_path+"plan_accuracy.txt", std::ios::out);
    std::ostream accuracy_f(&fb);
    //TODO: add env dimension in the vector
    accuracy_f << accuracy << "\n";
    fb.close();





    return 0;
}
