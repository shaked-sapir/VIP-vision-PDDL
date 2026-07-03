(define (domain gripper)
  (:requirements :strips :typing)
  (:types room ball gripper)

  (:predicates
    (at_robby ?r - room)
    (at ?b - ball ?r - room)
    (free ?g - gripper)
    (carry ?b - ball ?g - gripper)
  )

  (:action move
    :parameters (?from - room ?to - room)
    :precondition (and
      (at_robby ?from))
    :effect (and
      (not (at_robby ?from))
      (at_robby ?to))
  )

  (:action pick
    :parameters (?b - ball ?r - room ?g - gripper)
    :precondition (and
      (at_robby ?r)
      (at ?b ?r)
      (free ?g))
    :effect (and
      (not (at ?b ?r))
      (not (free ?g))
      (carry ?b ?g))
  )

  (:action drop
    :parameters (?b - ball ?r - room ?g - gripper)
    :precondition (and
      (at_robby ?r)
      (carry ?b ?g))
    :effect (and
      (at ?b ?r)
      (free ?g)
      (not (carry ?b ?g)))
  )
)
