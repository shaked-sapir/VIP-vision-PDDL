(define (domain blocks)
(:requirements :typing :equality :negative-preconditions :strips)
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
	:precondition (and (handempty ))
	:effect (and (holding ?x)
		(not (handempty )) 
		))

(:action put_down
	:parameters (?x - block)
	:precondition (and (holding ?x))
	:effect (and (clear ?x)
		(handempty )
		(not (holding ?x))
		(ontable ?x) 
		))

(:action stack
	:parameters (?x - block ?y - block)
	:precondition (and (holding ?x))
	:effect (and (clear ?x)
		(handempty )
		(not (clear ?y))
		(not (holding ?x))
		(not (ontable ?y))
		(on ?y ?x) 
		))

(:action unstack
	:parameters (?x - block ?y - block)
	:precondition (and (handempty ))
	:effect (and (holding ?x)
		(not (clear ?y))
		(not (handempty ))
		(not (on ?x ?y))
		(not (ontable ?y)) 
		))

)