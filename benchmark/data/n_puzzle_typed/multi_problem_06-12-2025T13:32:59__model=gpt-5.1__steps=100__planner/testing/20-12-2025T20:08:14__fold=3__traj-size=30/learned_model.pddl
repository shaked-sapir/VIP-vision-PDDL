(define (domain n_puzzle_typed)
(:requirements :equality :typing :negative-preconditions)
(:types 	position tile - object
)

(:predicates (at ?tile - tile ?position - position)
	(neighbor ?p1 - position ?p2 - position)
	(empty ?position - position)
)

(:action move
	:parameters (?tile - tile ?from - position ?to - position)
	:precondition (and )
	:effect (and (at ?tile ?from)
		(empty ?to)
		(not (neighbor ?from ?to))
		(not (neighbor ?to ?from)) 
		))

)