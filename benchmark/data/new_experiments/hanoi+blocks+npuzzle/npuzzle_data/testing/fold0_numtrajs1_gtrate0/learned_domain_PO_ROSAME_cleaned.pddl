(define (domain n_puzzle_typed)
(:requirements :typing)
(:types 	position tile - object
)

(:predicates (at ?tile - tile ?position - position)
	(neighbor ?p1 - position ?p2 - position)
	(empty ?position - position)
)

(:action move
	:parameters (?tile - tile ?from - position ?to - position)
	:precondition (and (at ?tile ?from))
	:effect (and  (not (at ?tile ?from))))

)