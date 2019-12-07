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

#include <iostream>
#include <fstream>
using namespace ompl;

int main()
{
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
