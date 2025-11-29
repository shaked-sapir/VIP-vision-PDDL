i want you to help me build an experiment system that runs pisam and noisy-pisam (with conflict search) against ROSAME (which i provided you the path to access it),
such that for each planning domain (blocks, hanoi, slidetile (AKA n_puzzle)) we build data and run experiments to report results. 
All relevant code should be placed in a new folder (inside this project) called "benchmark". code should be structured and compact as possible, if there is code unique to a specific domain you can 
choose to implement it an an ad-hoc folder for the domain inside the new folder, or in every other way you see fit. 
I will now guide you through the following stages, starting with **blocks** domain:

# equalize domain files
as ROSAME and our algorithms define the "blocks" domain file differently, we have to equalize them:

1. create an updated version of the blocks domain file, which is identical to that of ROSAME - without  the "handfull" predicate
2. create an updated system prompt (without the "handfull")

# generate data
now we have to generate images as learning data.

1. choose the first problem from pddlgym 
2. generate a visual trace of length 100 (i.e. a sequence of 100 images), save the full trace for rosame and for our algorithms cut it into 5 traces of length 15 (no overlap)
3. save the data in a folder structure that is easy to access for both osame and our algorithms

notice:
- you should also keep the actions taken at each step for both rosame and our algorithms (use literal.pddl_str() from obs.literals for this)
- actions are to be taken randomly in the problem, but must change the state (result in different images) - see the ImageTrajectoryHandler class for reference

# make some noise
now we have to add noise to the data for noisy-pisam and check rosame's robustness.

1. use LLMBlocksFluentClassifier to get predicates from images, save the full state (TRUE, FALSE, UNCERTAIN)
2. for pisam and noisy-pisam, save the .trajectory and .masking_info files as usual
3. for rosame, give each proposition a probability of being true, according to the following LLMBlocksFluentClassifier output:
   - TRUE -> 1
   - FALSE -> 0
   - UNCERTAIN -> 0.5

# run experiments
now we have to run the experiments.

## preparing problems
for this, use pddlgym's BlocksWorldEnv to get problems for testing (except for the problem you took to generate learning data from).
use:
- 5 problems from pddlgym/pddl/blocks
- 5 problems from pddlgym/pddl/blocks_test 
- 5 problems from pddlgym/pddl/blocks_medium

## running algorithms
1. run pisam and noisy-pisam with the generated data.
2. run rosame with the generated data.

## reporting results
1. for each algorithm, report the following metrics in a combined csv file for all algorithms for better comparison:
   - number of problems solved
   - number of problems timed out
   - number of problems failed
   - average planning time
   - average plan length

2. for each algorithm, report the following metrics in another combined csv file for all algorithms for better comparison:
   - learning time
   - data size used for learning
   - preconditions precision (compared to the ground truth domain model)
   - preconditions recall (compared to the ground truth domain model)
   - effects precision (compared to the ground truth domain model)
   - effects recall (compared to the ground truth domain model)

and also generate plots for visual comparison of the algorithms based on these metrics.

Show me your plan for implementing this.

