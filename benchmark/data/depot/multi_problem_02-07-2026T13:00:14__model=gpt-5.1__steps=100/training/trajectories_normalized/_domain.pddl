(define (domain depot)
  (:requirements :strips :typing)

  (:types
      depot
      truck
      crane
      pile
      package
  )

  (:predicates
      (at_truck ?t - truck ?d - depot)
      (at_crane ?c - crane ?d - depot)
      (at_pile ?pl - pile ?d - depot)

      (at ?p - package ?d - depot)

      (on ?p - package ?q - package)
      (on_pile ?p - package ?pl - pile)
      (clear ?x)

      (holding ?c - crane ?p - package)
      (empty_crane ?c - crane)

      (in_truck ?p - package ?t - truck)
  )

  (:action drive
    :parameters (?t - truck ?from - depot ?to - depot)
    :precondition (and 
        (at_truck ?t ?from))
    :effect (and
        (not (at_truck ?t ?from))
        (at_truck ?t ?to))
  )

  (:action lift
    :parameters (?c - crane ?p - package ?pl - pile ?d - depot)
    :precondition (and
        (at_crane ?c ?d)
        (at_pile ?pl ?d)
        (on_pile ?p ?pl)
        (at ?p ?d)
        (clear ?p)
        (empty_crane ?c))
    :effect (and
        (not (on_pile ?p ?pl))
        (holding ?c ?p)
        (not (clear ?p))
        (clear ?pl)
        (not (empty_crane ?c)))
  )

  (:action unstack
    :parameters (?c - crane ?p - package ?q - package ?d - depot)
    :precondition (and
        (at_crane ?c ?d)
        (on ?p ?q)
        (at ?p ?d)
        (at ?q ?d)
        (clear ?p)
        (empty_crane ?c))
    :effect (and
        (not (on ?p ?q))
        (holding ?c ?p)
        (clear ?q)
        (not (clear ?p))
        (not (empty_crane ?c)))
  )

  (:action drop
    :parameters (?c - crane ?p - package ?pl - pile ?d - depot)
    :precondition (and
        (at_crane ?c ?d)
        (at_pile ?pl ?d)
        (holding ?c ?p))
    :effect (and
        (on_pile ?p ?pl)
        (at ?p ?d)
        (clear ?p)
        (empty_crane ?c)
        (not (holding ?c ?p)))
  )

  (:action stack
    :parameters (?c - crane ?p - package ?q - package ?d - depot)
    :precondition (and
        (at_crane ?c ?d)
        (holding ?c ?p)
        (clear ?q)
        (at ?q ?d))
    :effect (and
        (on ?p ?q)
        (at ?p ?d)
        (clear ?p)
        (empty_crane ?c)
        (not (holding ?c ?p))
        (not (clear ?q)))
  )

  (:action load
    :parameters (?c - crane ?p - package ?t - truck ?d - depot)
    :precondition (and
        (at_crane ?c ?d)
        (at_truck ?t ?d)
        (holding ?c ?p))
    :effect (and
        (in_truck ?p ?t)
        (not (at ?p ?d))
        (empty_crane ?c)
        (not (holding ?c ?p)))
  )

  (:action unload
    :parameters (?c - crane ?p - package ?t - truck ?d - depot)
    :precondition (and
        (at_crane ?c ?d)
        (at_truck ?t ?d)
        (in_truck ?p ?t)
        (empty_crane ?c))
    :effect (and
        (holding ?c ?p)
        (at ?p ?d)
        (not (in_truck ?p ?t))
        (not (empty_crane ?c)))
  )
)
