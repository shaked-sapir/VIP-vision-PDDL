(define (domain hiking)
(:requirements :equality :typing :strips :negative-preconditions)
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
	:precondition (and )
	:effect (and (at ?to)
		(not (ishill ?from)) 
		))

)