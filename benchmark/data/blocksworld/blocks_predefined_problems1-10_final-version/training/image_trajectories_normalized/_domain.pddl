;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; 4 op_blocks world, matching pddlgym's blocks_operator_actions
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

; This domain matches the post_transform format produced by the LLM/contour
; classifiers and the trajectory handler: underscore action names (pick_up,
; put_down) and a nullary (handempty) with no (handfull). It is the single
; canonical domain used for BOTH data creation (inference/masking) and
; benchmark evaluation.

(define (domain blocks)
    (:requirements :strips :typing)
    (:types block)
    (:predicates
        (on ?x - block ?y - block)
        (ontable ?x - block)
        (clear ?x - block)
        (handempty)
        (holding ?x - block)
    )

    ; (:actions pick_up put_down stack unstack)

    (:action pick_up
        :parameters (?x - block)
        :precondition (and
            (clear ?x)
            (ontable ?x)
            (handempty)
        )
        :effect (and
            (not (ontable ?x))
            (not (clear ?x))
            (not (handempty))
            (holding ?x)
        )
    )

    (:action put_down
        :parameters (?x - block)
        :precondition (and
            (holding ?x)
        )
        :effect (and
            (not (holding ?x))
            (clear ?x)
            (handempty)
            (ontable ?x))
        )

    (:action stack
        :parameters (?x - block ?y - block)
        :precondition (and
            (holding ?x)
            (clear ?y)
        )
        :effect (and
            (not (holding ?x))
            (not (clear ?y))
            (clear ?x)
            (handempty)
            (on ?x ?y)
        )
    )

    (:action unstack
        :parameters (?x - block ?y - block)
        :precondition (and
            (on ?x ?y)
            (clear ?x)
            (handempty)
        )
        :effect (and
            (holding ?x)
            (clear ?y)
            (not (clear ?x))
            (not (handempty))
            (not (on ?x ?y))
        )
    )
)
