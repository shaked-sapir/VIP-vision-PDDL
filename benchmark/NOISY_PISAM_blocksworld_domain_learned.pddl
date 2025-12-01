(define (domain blocksworld)
(:requirements :strips :typing :negative-preconditions :equality)
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
	:effect (and (not (handempty )) 
		))

(:action put_down
	:parameters (?x - block)
	:precondition (and )
	:effect (and (handempty ) 
		))

(:action stack
	:parameters (?x - block ?y - block)
	:precondition (and )
	:effect (and (handempty )
		(not (holding ?y)) 
		))

(:action unstack
	:parameters (?x - block ?y - block)
	:precondition (and (handempty ))
	:effect (and (holding ?y)
		(not (clear ?y))
		(not (handempty )) 
		))

)