(define (domain hanoi)
(:requirements :typing :strips)
(:types 	peg disc - object
)

(:predicates (clear-peg ?x - peg)
	(clear-disc ?x - disc)
	(on-disc ?x - disc ?y - disc)
	(on-peg ?x - disc ?y - peg)
	(smaller-disc ?x - disc ?y - disc)
	(smaller-peg ?x - peg ?y - disc)
)

(:action move_disc_disc
	:parameters (?disc - disc ?from - disc ?to - disc)
	:precondition (and (clear-disc ?disc) (clear-disc ?from) (on-disc ?from ?disc) (on-disc ?from ?to) (on-disc ?to ?from) (smaller-disc ?from ?disc) (smaller-disc ?from ?to) (smaller-disc ?to ?from))
	:effect (and (clear-disc ?to) (on-disc ?disc ?from) (on-disc ?disc ?to) (on-disc ?to ?disc) (smaller-disc ?disc ?from) (smaller-disc ?disc ?to) (smaller-disc ?to ?disc)))

(:action move_disc_peg
	:parameters (?disc - disc ?from - disc ?to - peg)
	:precondition (and (on-disc ?disc ?from))
	:effect (and (clear-disc ?disc) (clear-disc ?from) (on-disc ?from ?disc) (on-peg ?from ?to) (smaller-disc ?disc ?from) (smaller-disc ?from ?disc) (smaller-peg ?to ?disc) (smaller-peg ?to ?from) (not (on-disc ?disc ?from))))

(:action move_peg_disc
	:parameters (?disc - disc ?from - peg ?to - disc)
	:precondition (and (clear-disc ?to) (on-peg ?disc ?from))
	:effect (and (clear-peg ?from) (on-disc ?disc ?to) (on-disc ?to ?disc) (on-peg ?to ?from) (smaller-disc ?disc ?to) (not (clear-disc ?to))  (not (on-peg ?disc ?from))))

(:action move_peg_peg
	:parameters (?disc - disc ?from - peg ?to - peg)
	:precondition (and )
	:effect (and (clear-peg ?from) (clear-peg ?to) (clear-disc ?disc) (on-peg ?disc ?from) (on-peg ?disc ?to) (smaller-peg ?from ?disc) (smaller-peg ?to ?disc)))

)