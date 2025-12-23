(define (domain hanoi)
(:requirements :typing :strips :negative-preconditions :equality)
(:types 	peg disc - object
)

(:predicates (clear-peg ?x - peg)
	(clear-disc ?x - disc)
	(on-disc ?x - disc ?y - disc)
	(on-peg ?x - disc ?y - peg)
	(smaller-disc ?x - disc ?y - disc)
	(smaller-peg ?x - peg ?y - disc)
)

(:action move_disc_peg
	:parameters (?disc - disc ?from - disc ?to - peg)
	:precondition (and (clear-disc ?disc)
	(clear-peg ?to)
	(on-disc ?disc ?from)
	(smaller-disc ?from ?disc)
	(smaller-peg ?to ?disc)
	(smaller-peg ?to ?from))
	:effect (and (clear-disc ?from)
		(on-peg ?disc ?to) 
		))

)