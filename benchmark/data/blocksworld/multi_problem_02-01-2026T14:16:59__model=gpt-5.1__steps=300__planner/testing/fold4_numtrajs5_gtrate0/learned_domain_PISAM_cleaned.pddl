(define (domain blocks)
(:requirements :typing :strips :negative-preconditions :equality)
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
	:effect (and (clear ?x)
		(holding ?x)
		(not (handempty )) 
		))

(:action put_down
	:parameters (?x - block)
	:precondition (and (holding ?x))
	:effect (and (handempty )
		(not (holding ?x)) 
		))

(:action stack
	:parameters (?x - block ?y - block)
	:precondition (and (holding ?x))
	:effect (and (clear ?x)
		(handempty )
		(not (holding ?x)) 
		))

(:action unstack
	:parameters (?x - block ?y - block)
	:precondition (and (handempty ))
	:effect (and (holding ?x)
		(not (handempty )) 
		))

)