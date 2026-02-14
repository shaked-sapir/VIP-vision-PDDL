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
	:precondition (and (at ?tile ?from) (neighbor ?from ?to) (empty ?to))
	:effect (and (at ?tile ?to) (neighbor ?to ?from) (empty ?from) (not (at ?tile ?from))  (not (empty ?to))))

)