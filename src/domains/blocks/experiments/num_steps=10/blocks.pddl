;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; 4 op-blocks world, same as in pddlgym
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

; in the original file I used, there were 4 more predicates:
; (pickup ?x - block) // whether the block can be picked up in general, without considering the other predicates (e.g. it is super-heavy..)
; (putdown ?x - block) // whether the block can be put down in general, without considering the other predicates (e.g. it is super-heavy..)
; (stack ?x - block ?y - block) // whether the block can be stacked on another block in general, without considering the other predicates (e.g. we cannot put big box on small box)
; (unstack ?x - block) // whether the block can be unstacked in general, without considering the other predicates (e.g. it is too sensitive, requiring human intervention)

; I decided to remove them because in our implementation we do not use them, and they are not needed for
; the blocks world to work (they are always true), making the states more verbose without adding any value.

; so I also removed them from the preconditions of all actions - each was a precondition to its corresponding action,
; for example (pickup ?x) was a precondition of the pick-up action, and it was always true.

(define (domain blocks)
    (:requirements :strips :typing)
    (:types block robot)
    (:predicates
        (on ?x - block ?y - block)
        (ontable ?x - block)
        (clear ?x - block)
        (handempty ?x - robot)
        (handfull ?x - robot)
        (holding ?x - block)
    )

    ; (:actions pick-up put-down stack unstack)

    (:action pick-up
        :parameters (?x - block ?robot - robot)
        :precondition (and
            (clear ?x)
            (ontable ?x)
            (handempty ?robot)
        )
        :effect (and
            (not (ontable ?x))
            (not (clear ?x))
            (not (handempty ?robot))
            (handfull ?robot)
            (holding ?x)
        )
    )

    (:action put-down
        :parameters (?x - block ?robot - robot)
        :precondition (and
            (holding ?x)
            (handfull ?robot)
        )
        :effect (and
            (not (holding ?x))
            (clear ?x)
            (handempty ?robot)
            (not (handfull ?robot))
            (ontable ?x))
        )

    (:action stack
        :parameters (?x - block ?y - block ?robot - robot)
        :precondition (and
            (holding ?x)
            (clear ?y)
            (handfull ?robot)
        )
        :effect (and
            (not (holding ?x))
            (not (clear ?y))
            (clear ?x)
            (handempty ?robot)
            (not (handfull ?robot))
            (on ?x ?y)
        )
    )

    (:action unstack
        :parameters (?x - block ?y - block ?robot - robot)
        :precondition (and
            (on ?x ?y)
            (clear ?x)
            (handempty ?robot)
        )
        :effect (and
            (holding ?x)
            (clear ?y)
            (not (clear ?x))
            (not (handempty ?robot))
            (handfull ?robot)
            (not (on ?x ?y))
        )
    )
)