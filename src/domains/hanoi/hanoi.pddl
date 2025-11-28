(define (domain hanoi)
    (:requirements :strips)
    (:types peg disc)
    (:predicates
        (clear-peg ?x - peg) ; means that the peg ?x is clear, i.e., nothing is on it
        (clear-disc ?x - disc) ; means that the disc ?x is clear, i.e., nothing is on it
        (on-disc ?x - disc ?y - disc) ; means that the disc ?x is on the disc ?y
        (on-peg ?x - disc ?y - peg) ; means that the disc ?x is on the peg ?y
        (smaller-disc ?x - disc ?y - disc) ; means that the disc ?y is smaller than the disc ?x
        (smaller-peg ?x - peg ?y - disc) ; means that the disc ?y is smaller than the peg ?x
    )

    (:action move-disc-disc
        :parameters (?disc - disc ?from - disc ?to - disc)
        :precondition (and
            (smaller-disc ?to ?disc)
            (on-disc ?disc ?from)
            (clear-disc ?disc)
            (clear-disc ?to)
        )
        :effect  (and
            (clear-disc ?from)
            (on-disc ?disc ?to)
            (not (on-disc ?disc ?from))
            (not (clear-disc ?to))
        )
    )

    (:action move-disc-peg
        :parameters (?disc - disc ?from - disc ?to - peg)
        :precondition (and
            (smaller-peg ?to ?disc)
            (on-disc ?disc ?from)
            (clear-disc ?disc)
            (clear-peg ?to)
        )
        :effect  (and
            (clear-disc ?from)
            (on-peg ?disc ?to)
            (not (on-disc ?disc ?from))
            (not (clear-peg ?to))
        )
    )

    (:action move-peg-disc
        :parameters (?disc - disc ?from - peg ?to - disc)
        :precondition (and
            (smaller-disc ?to ?disc)
            (on-peg ?disc ?from)
            (clear-disc ?disc)
            (clear-disc ?to)
        )
        :effect  (and
            (clear-peg ?from)
            (on-disc ?disc ?to)
            (not (on-peg ?disc ?from))
            (not (clear-disc ?to))
        )
    )

    (:action move-peg-peg
        :parameters (?disc - disc ?from - peg ?to - peg)
        :precondition (and
            (smaller-peg ?to ?disc)
            (on-peg ?disc ?from)
            (clear-disc ?disc)
            (clear-peg ?to)
        )
        :effect  (and
            (clear-peg ?from)
            (on-peg ?disc ?to)
            (not (on-peg ?disc ?from))
            (not (clear-peg ?to))
        )
    )
)