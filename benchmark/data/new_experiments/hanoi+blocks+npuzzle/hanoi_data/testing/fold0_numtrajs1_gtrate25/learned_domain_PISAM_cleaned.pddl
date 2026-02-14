(define (domain hanoi)
(:requirements :equality :negative-preconditions :typing :strips)
(:types 	peg disc - object
)

(:predicates (clear-peg ?x - peg)
	(clear-disc ?x - disc)
	(on-disc ?x - disc ?y - disc)
	(on-peg ?x - disc ?y - peg)
	(smaller-disc ?x - disc ?y - disc)
	(smaller-peg ?x - peg ?y - disc)
)

(:action move_peg_disc
	:parameters (?disc - disc ?from - peg ?to - disc)
	:precondition (and (clear-disc ?disc)
	(clear-disc ?to)
	(on-peg ?disc ?from)
	(smaller-disc ?to ?disc)
	(smaller-peg ?from ?disc)
	(smaller-peg ?from ?to))
	:effect (and (clear-peg ?from)
		(not (clear-disc ?to))
		(on-disc ?disc ?to) 
		))

)