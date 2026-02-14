(define (domain hanoi)
(:requirements :strips :typing)
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
	:precondition (and (clear-disc ?disc) (on-disc ?disc ?from) (on-disc ?disc ?to) (on-disc ?to ?from) (smaller-disc ?disc ?to) (smaller-disc ?from ?disc) (smaller-disc ?from ?to) (smaller-disc ?to ?disc) (smaller-disc ?to ?from))
	:effect (and (clear-disc ?from) (clear-disc ?to) (on-disc ?from ?disc) (on-disc ?from ?to) (on-disc ?to ?disc) (smaller-disc ?disc ?from) (not (on-disc ?to ?from))  (not (smaller-disc ?disc ?to))))

(:action move_disc_peg
	:parameters (?disc - disc ?from - disc ?to - peg)
	:precondition (and (on-disc ?disc ?from) (smaller-disc ?from ?disc) (smaller-peg ?to ?disc) (smaller-peg ?to ?from))
	:effect (and (clear-disc ?disc) (clear-disc ?from) (on-disc ?from ?disc) (on-peg ?disc ?to) (on-peg ?from ?to) (smaller-disc ?disc ?from) (not (on-disc ?disc ?from))))

(:action move_peg_disc
	:parameters (?disc - disc ?from - peg ?to - disc)
	:precondition (and (clear-disc ?disc) (on-disc ?disc ?to) (on-disc ?to ?disc) (on-peg ?disc ?from) (smaller-disc ?disc ?to) (smaller-disc ?to ?disc) (smaller-peg ?from ?disc) (smaller-peg ?from ?to))
	:effect (and (clear-disc ?to) (not (on-disc ?to ?disc))  (not (on-peg ?disc ?from))  (not (smaller-disc ?disc ?to))))

(:action move_peg_peg
	:parameters (?disc - disc ?from - peg ?to - peg)
	:precondition (and (smaller-peg ?from ?disc) (smaller-peg ?to ?disc))
	:effect (and (clear-peg ?from) (on-peg ?disc ?to)))

)