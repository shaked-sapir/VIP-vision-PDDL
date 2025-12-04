(define (domain blocks)
(:requirements :strips :equality :negative-preconditions :typing)
(:types 	block - object
)

(:predicates (on ?x - block ?y - block)
	(ontable ?x - block)
	(clear ?x - block)
	(handempty )
	(holding ?x - block)
)

(:action pick_up
	:parameters (?x - block)
	:precondition (and (clear ?x)
	(ontable ?x))
	:effect (and (holding ?x)
		(not (clear ?x))
		(not (handempty ))
		(not (ontable ?x)) 
		))

(:action stack
	:parameters (?x - block ?y - block)
	:precondition (and (clear ?y)
	(holding ?x))
	:effect (and (clear ?x)
		(not (clear ?y))
		(not (holding ?x)) 
		))

)