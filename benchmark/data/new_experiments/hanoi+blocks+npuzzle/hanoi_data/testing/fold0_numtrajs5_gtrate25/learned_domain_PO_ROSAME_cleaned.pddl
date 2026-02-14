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
	:precondition (and (clear-disc ?disc) (clear-disc ?to) (on-disc ?disc ?from) (smaller-disc ?from ?disc) (smaller-disc ?to ?disc) (smaller-disc ?to ?from))
	:effect (and (clear-disc ?from) (on-disc ?disc ?to) (on-disc ?from ?disc) (on-disc ?from ?to) (on-disc ?to ?disc) (on-disc ?to ?from) (smaller-disc ?disc ?from) (smaller-disc ?disc ?to) (smaller-disc ?from ?to) (not (clear-disc ?to))  (not (on-disc ?disc ?from))))

(:action move_disc_peg
	:parameters (?disc - disc ?from - disc ?to - peg)
	:precondition (and (clear-peg ?to) (on-disc ?disc ?from) (smaller-disc ?disc ?from) (smaller-disc ?from ?disc) (smaller-peg ?to ?disc) (smaller-peg ?to ?from))
	:effect (and (clear-disc ?disc) (clear-disc ?from) (on-peg ?disc ?to) (not (clear-peg ?to))  (not (on-disc ?disc ?from))  (not (smaller-disc ?disc ?from))))

(:action move_peg_disc
	:parameters (?disc - disc ?from - peg ?to - disc)
	:precondition (and (clear-disc ?disc) (clear-disc ?to) (on-disc ?to ?disc) (on-peg ?disc ?from) (on-peg ?to ?from) (smaller-disc ?disc ?to) (smaller-disc ?to ?disc) (smaller-peg ?from ?disc) (smaller-peg ?from ?to))
	:effect (and (clear-peg ?from) (on-disc ?disc ?to) (not (clear-disc ?to))  (not (on-disc ?to ?disc))  (not (on-peg ?disc ?from))  (not (on-peg ?to ?from))  (not (smaller-disc ?disc ?to))))

(:action move_peg_peg
	:parameters (?disc - disc ?from - peg ?to - peg)
	:precondition (and (clear-peg ?to) (clear-disc ?disc) (on-peg ?disc ?from) (smaller-peg ?from ?disc) (smaller-peg ?to ?disc))
	:effect (and (clear-peg ?from) (on-peg ?disc ?to) (not (clear-peg ?to))  (not (on-peg ?disc ?from))))

)