from pddlgym.parser import Operator

"""
    At the moment, this class implements the exact same logic as PDDLGym's Operator. 
    It includes signals for preconditions, params and effects of the proper action (operator) schema.
    Note: we need to be able to distinguish between ADD and DEL effects among the schema's effects.
          I think we could consider a positive literal result as an ADD and to negative literal result as a DEL.
"""
class Action(Operator):
    pass
