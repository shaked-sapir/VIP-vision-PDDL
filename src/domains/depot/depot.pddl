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
      (at-truck ?t - truck ?d - depot)
      (at-crane ?c - crane ?d - depot)
      (at-pile ?pl - pile ?d - depot)

      (at ?p - package ?d - depot)

      (on ?p - package ?q - package)
      (on-pile ?p - package ?pl - pile)
      (clear ?x)

      (holding ?c - crane ?p - package)
      (empty-crane ?c - crane)

      (in-truck ?p - package ?t - truck)
  )

  (:action drive
    :parameters (?t - truck ?from - depot ?to - depot)
    :precondition (at-truck ?t ?from)
    :effect (and
        (not (at-truck ?t ?from))
        (at-truck ?t ?to))
  )

  (:action lift
    :parameters (?c - crane ?p - package ?pl - pile ?d - depot)
    :precondition (and
        (at-crane ?c ?d)
        (at-pile ?pl ?d)
        (on-pile ?p ?pl)
        (at ?p ?d)
        (clear ?p)
        (empty-crane ?c))
    :effect (and
        (not (on-pile ?p ?pl))
        (holding ?c ?p)
        (not (clear ?p))
        (clear ?pl)
        (not (empty-crane ?c)))
  )

  (:action unstack
    :parameters (?c - crane ?p - package ?q - package ?d - depot)
    :precondition (and
        (at-crane ?c ?d)
        (on ?p ?q)
        (at ?p ?d)
        (at ?q ?d)
        (clear ?p)
        (empty-crane ?c))
    :effect (and
        (not (on ?p ?q))
        (holding ?c ?p)
        (clear ?q)
        (not (clear ?p))
        (not (empty-crane ?c)))
  )

  (:action drop
    :parameters (?c - crane ?p - package ?pl - pile ?d - depot)
    :precondition (and
        (at-crane ?c ?d)
        (at-pile ?pl ?d)
        (holding ?c ?p))
    :effect (and
        (on-pile ?p ?pl)
        (at ?p ?d)
        (clear ?p)
        (empty-crane ?c)
        (not (holding ?c ?p)))
  )

  (:action stack
    :parameters (?c - crane ?p - package ?q - package ?d - depot)
    :precondition (and
        (at-crane ?c ?d)
        (holding ?c ?p)
        (clear ?q)
        (at ?q ?d))
    :effect (and
        (on ?p ?q)
        (at ?p ?d)
        (clear ?p)
        (empty-crane ?c)
        (not (holding ?c ?p))
        (not (clear ?q)))
  )

  (:action load
    :parameters (?c - crane ?p - package ?t - truck ?d - depot)
    :precondition (and
        (at-crane ?c ?d)
        (at-truck ?t ?d)
        (holding ?c ?p))
    :effect (and
        (in-truck ?p ?t)
        (not (at ?p ?d))
        (empty-crane ?c)
        (not (holding ?c ?p)))
  )

  (:action unload
    :parameters (?c - crane ?p - package ?t - truck ?d - depot)
    :precondition (and
        (at-crane ?c ?d)
        (at-truck ?t ?d)
        (in-truck ?p ?t)
        (empty-crane ?c))
    :effect (and
        (holding ?c ?p)
        (at ?p ?d)
        (not (in-truck ?p ?t))
        (not (empty-crane ?c)))
  )
)
