(define (domain hanoi)
(:requirements :typing :strips :equality :negative-preconditions)
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
	:precondition (and (clear-peg ?to)
	(on-disc ?disc ?from)
	(smaller-disc ?from ?disc)
	(smaller-peg ?to ?disc)
	(smaller-peg ?to ?from))
	:effect (and (clear-disc ?disc)
		(not (clear-peg ?to))
		(on-peg ?disc ?to) 
		))

(:action move_peg_disc
	:parameters (?disc - disc ?from - peg ?to - disc)
	:precondition (and (clear-disc ?disc)
	(clear-disc ?to)
	(on-peg ?disc ?from)
	(smaller-disc ?to ?disc)
	(smaller-peg ?from ?disc)
	(smaller-peg ?from ?to))
	:effect (and (clear-peg ?from)
		(not (on-peg ?disc ?from))
		(on-disc ?disc ?to) 
		))

)