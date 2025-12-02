(define (domain hiking)
(:requirements :typing :strips)
(:types 	loc - object
)

(:predicates (at ?loc - loc)
	(iswater ?loc - loc)
	(ishill ?loc - loc)
	(isgoal ?loc - loc)
	(adjacent ?loc1 - loc ?loc2 - loc)
)

(:action walk
	:parameters (?from - loc ?to - loc)
	:precondition (and (adjacent ?from ?to) (adjacent ?to ?from))
	:effect (and ))

(:action climb
	:parameters (?from - loc ?to - loc)
	:precondition (and (at ?from) (at ?to) (iswater ?from) (iswater ?to) (ishill ?from) (ishill ?to) (isgoal ?from) (isgoal ?to) (adjacent ?from ?to) (adjacent ?to ?from))
	:effect (and ))

)