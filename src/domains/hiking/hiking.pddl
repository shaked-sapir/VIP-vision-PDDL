(define (domain hiking)
    (:requirements :strips :typing)
    (:types loc)

    (:predicates
      (at ?loc - loc)
      (iswater ?loc - loc)
      (ishill ?loc - loc)
      (isgoal ?loc - loc)
      (adjacent ?loc1 - loc ?loc2 - loc)
    )

   ; (:actions walk climb)


    (:action walk
      :parameters (?from - loc ?to - loc)
      :precondition (and
        (not (ishill ?to))
        (at ?from)
        (adjacent ?from ?to)
        (not (iswater ?from)))
      :effect (and (at ?to) (not (at ?from)))
    )

    (:action climb
      :parameters (?from - loc ?to - loc)
      :precondition (and
        (ishill ?to)
        (at ?from)
        (adjacent ?from ?to)
        (not (iswater ?from)))
      :effect (and (at ?to) (not (at ?from)))
    )
)