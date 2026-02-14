(define (domain n_puzzle_typed)
(:requirements :typing :negative-preconditions :equality)
(:types 	position tile - object
)

(:predicates (at ?tile - tile ?position - position)
	(neighbor ?p1 - position ?p2 - position)
	(empty ?position - position)
)

(:action move
	:parameters (?tile - tile ?from - position ?to - position)
	:precondition (and )
	:effect (and (at ?tile ?to)
		(empty ?from)
		(not (at ?tile ?from))
		(not (empty ?to)) 
		))

)