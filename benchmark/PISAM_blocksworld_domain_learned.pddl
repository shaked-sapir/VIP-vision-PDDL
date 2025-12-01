(define (domain blocksworld)
(:requirements :negative-preconditions :strips :equality :typing)
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
	:effect (and (clear ?x)
		(clear ?y)
		(handempty )
		(not (clear ?x))
		(not (clear ?y))
		(not (holding ?x))
		(not (holding ?y)) 
		))

(:action unstack
	:parameters (?x - block ?y - block)
	:precondition (and (handempty ))
	:effect (and (clear ?x)
		(clear ?y)
		(holding ?x)
		(holding ?y)
		(not (clear ?x))
		(not (clear ?y))
		(not (handempty )) 
		))

)