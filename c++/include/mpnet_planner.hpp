#ifndef OMPL_PLANNERS_MPNET_
#define OMPL_PLANNERS_MPNET_

#include "ompl/geometric/planners/PlannerIncludes.h"
#include <torch/torch.h>
#include <torch/script.h>


using namespace ompl;
typedef std::vector<ompl::base::State *> StatePtrVec;

class MPNetPlanner : public base::Planner
{
public:
    /** \brief Constructor */
    MPNetPlanner(const base::SpaceInformationPtr &si, bool addIntermediateStates = false, int max_replan = 1001, int max_length = 3000);

    ~MPNetPlanner() override;

    void getPlannerData(base::PlannerData &data) const override;

    base::PlannerStatus solve(const base::PlannerTerminationCondition &ptc) override;

    void clear() override;

    /** \brief Set the goal bias
        In the process of randomly selecting states in
        the state space to attempt to go towards, the
        algorithm may in fact choose the actual goal state, if
        it knows it, with some probability. This probability
        is a real number between 0.0 and 1.0; its value should
        usually be around 0.05 and should not be too large. It
        is probably a good idea to use the default value. */
    void setGoalBias(double goalBias)
    {
        goalBias_ = goalBias;
    }

    /** \brief Get the goal bias the planner is using */
    double getGoalBias() const
    {
        return goalBias_;
    }

    /** \brief Return true if the intermediate states generated along motions are to be added to the tree itself
     */
    bool getIntermediateStates() const
    {
        return addIntermediateStates_;
    }

    /** \brief Specify whether the intermediate states generated along motions are to be added to the tree
     * itself */
    void setIntermediateStates(bool addIntermediateStates)
    {
        addIntermediateStates_ = addIntermediateStates;
    }

    /** \brief Set the range the planner is supposed to use.
        This parameter greatly influences the runtime of the
        algorithm. It represents the maximum length of a
        motion to be added in the tree of motions. */
    void setRange(double distance)
    {
        maxDistance_ = distance;
    }

    /** \brief Get the range the planner is using */
    double getRange() const
    {
        return maxDistance_;
    }

    void setup() override;

protected:
    /** \brief Representation of a motion
        This only contains pointers to parent motions as we
        only need to go backwards in the tree. */
    at::Tensor obs_enc; // two dimensional or one dimensional
    std::shared_ptr<torch::jit::script::Module> encoder;
    std::shared_ptr<torch::jit::script::Module> MLP;
    std::vector<float> lower_bound = {-383.8, -371.47, -0.2};
    std::vector<float> upper_bound = {325, 337.89, 142.33};
    std::vector<float> bound = {0., 0., 0.};
    // MPNet specific:
    StatePtrVec neural_replan(StatePtrVec path, int max_length);
    StatePtrVec neural_replanner(base::State* start, base::State* goal, int max_length);
    virtual std::vector<float> normalize(std::vector<float> state, int dim);
    virtual std::vector<float> unnormalize(std::vector<float> state, int dim);
    void mpnet_predict(const base::State* start, const base::State* goal, base::State* next);
    torch::Tensor getStartGoalTensor(const base::State *start_state, const base::State *goal_state, int dim);
    StatePtrVec lvc(StatePtrVec path);

    class Motion
    {
    public:
        Motion() = default;

        /** \brief Constructor that allocates memory for the state */
        Motion(const base::SpaceInformationPtr &si) : state(si->allocState())
        {
        }

        ~Motion() = default;

        /** \brief The state contained by the motion */
        base::State *state{nullptr};

        /** \brief The parent motion in the exploration tree */
        Motion *parent{nullptr};
    };

    /** \brief Free the memory allocated by this planner */
    void freeMemory();

    /** \brief Compute distance between motions (actually distance between contained states) */
    double distanceFunction(const Motion *a, const Motion *b) const
    {
        return si_->distance(a->state, b->state);
    }

    /** \brief State sampler */
    base::StateSamplerPtr sampler_;

    /** \brief The fraction of time the goal is picked as the state to expand towards (if such a state is
     * available) */
    double goalBias_{.05};

    /** \brief The maximum length of a motion to be added to a tree */
    double maxDistance_{0.};

    /** \brief Flag indicating whether intermediate states are added to the built tree of motions */
    bool addIntermediateStates_;

    /** \brief The random number generator */
    RNG rng_;

    /** \brief The most recent goal motion.  Used for PlannerData computation */
    Motion *lastGoalMotion_{nullptr};

};

#endif
