(define (domain blocks)
(:requirements :equality :strips :typing :negative-preconditions)
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
	:precondition (and (handempty )
	(ontable ?x))
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
		(clear ?y)
		(handempty )
		(not (holding ?x))
		(not (ontable ?y))
		(on ?x ?y) 
		))

(:action unstack
	:parameters (?x - block ?y - block)
	:precondition (and (handempty ))
	:effect (and (holding ?x)
		(not (clear ?x))
		(not (handempty ))
		(not (on ?x ?y))
		(not (on ?y ?x))
		(not (ontable ?x))
		(ontable ?y) 
		))

)