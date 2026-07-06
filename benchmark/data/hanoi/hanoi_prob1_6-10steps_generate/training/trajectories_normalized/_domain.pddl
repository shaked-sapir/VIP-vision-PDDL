(define (domain hanoi)
    (:requirements :strips)
    (:types peg disc)
    (:predicates
        (clear_peg ?x - peg) ; means that the peg ?x is clear, i.e., nothing is on it
        (clear_disc ?x - disc) ; means that the disc ?x is clear, i.e., nothing is on it
        (on_disc ?x - disc ?y - disc) ; means that the disc ?x is on the disc ?y
        (on_peg ?x - disc ?y - peg) ; means that the disc ?x is on the peg ?y
        (smaller_disc ?x - disc ?y - disc) ; means that the disc ?y is smaller than the disc ?x
        (smaller_peg ?x - peg ?y - disc) ; means that the disc ?y is smaller than the peg ?x
    )

    (:action move_disc_disc
        :parameters (?disc - disc ?from - disc ?to - disc)
        :precondition (and
            (smaller_disc ?to ?disc)
            (on_disc ?disc ?from)
            (clear_disc ?disc)
            (clear_disc ?to)
        )
        :effect  (and
            (clear_disc ?from)
            (on_disc ?disc ?to)
            (not (on_disc ?disc ?from))
            (not (clear_disc ?to))
        )
    )

    (:action move_disc_peg
        :parameters (?disc - disc ?from - disc ?to - peg)
        :precondition (and
            (smaller_peg ?to ?disc)
            (on_disc ?disc ?from)
            (clear_disc ?disc)
            (clear_peg ?to)
        )
        :effect  (and
            (clear_disc ?from)
            (on_peg ?disc ?to)
            (not (on_disc ?disc ?from))
            (not (clear_peg ?to))
        )
    )

    (:action move_peg_disc
        :parameters (?disc - disc ?from - peg ?to - disc)
        :precondition (and
            (smaller_disc ?to ?disc)
            (on_peg ?disc ?from)
            (clear_disc ?disc)
            (clear_disc ?to)
        )
        :effect  (and
            (clear_peg ?from)
            (on_disc ?disc ?to)
            (not (on_peg ?disc ?from))
            (not (clear_disc ?to))
        )
    )

    (:action move_peg_peg
        :parameters (?disc - disc ?from - peg ?to - peg)
        :precondition (and
            (smaller_peg ?to ?disc)
            (on_peg ?disc ?from)
            (clear_disc ?disc)
            (clear_peg ?to)
        )
        :effect  (and
            (clear_peg ?from)
            (on_peg ?disc ?to)
            (not (on_peg ?disc ?from))
            (not (clear_peg ?to))
        )
    )
)