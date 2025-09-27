(define (domain hanoi)
  (:requirements :strips)
  (:predicates
  (clear ?x) ; means that the disc/peg ?x is clear, i.e., nothing is on it
  (on ?x ?y) ; means that the disc ?x is on the disc/peg ?y
  (smaller ?x ?y) ; means that the disc ?y is smaller than the disc/peg ?x
  )

  (:action move
    :parameters (?disc ?from ?to)
    :precondition (and (smaller ?to ?disc) (on ?disc ?from)
               (clear ?disc) (clear ?to))
    :effect  (and (clear ?from) (on ?disc ?to) (not (on ?disc ?from))
          (not (clear ?to))))
  )
