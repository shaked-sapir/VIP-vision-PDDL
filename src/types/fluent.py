class Fluent:
    """
    This class represents a logic Fluent, which is a grounded predicate, i.e. contain specific object for each type
    in the predicate.

    For example, if the domain contains a predicate  "On(<Cube>1, <Cube>2)", meaning Cube1 is placed on top of Cube2,
    and the objects of type Cube in the current problem (in the domain) are c1, c2, c3 - then On(c1, c2) and On(c2, c3)
    are valid fluents in the problem. (meaning that c1 is placed on c2 and c2 is placed on c3)
    """
    pass