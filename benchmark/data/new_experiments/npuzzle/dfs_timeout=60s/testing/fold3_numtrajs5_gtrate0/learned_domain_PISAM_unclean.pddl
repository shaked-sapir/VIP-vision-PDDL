(define (domain n_puzzle_typed)
(:requirements :negative-preconditions :equality :typing)
(:types 	position tile - object
)

(:predicates (at ?tile - tile ?position - position)
	(neighbor ?p1 - position ?p2 - position)
	(empty ?position - position)
)

(:action move
	:parameters (?tile - tile ?from - position ?to - position)
	:precondition (and (at ?tile ?from)
	(neighbor ?from ?to)
	(neighbor ?to ?from))
	:effect (and (at ?tile ?to)
		(empty ?from)
		(not (at ?tile ?from))
		(not (empty ?to))
		(not (neighbor ?from ?to))
		(not (neighbor ?to ?from)) 
		))

)