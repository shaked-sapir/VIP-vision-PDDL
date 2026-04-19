(define (domain blocks)
(:requirements :strips :equality :typing :negative-preconditions)
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
	(handempty ))
	:effect (and (holding ?x)
		(not (clear ?x))
		(not (handempty ))
		(not (ontable ?x)) 
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
	:effect (and (handempty )
		(not (clear ?y))
		(not (holding ?x))
		(not (ontable ?y))
		(on ?x ?y)
		(on ?y ?x) 
		))

(:action unstack
	:parameters (?x - block ?y - block)
	:precondition (and (clear ?x)
	(handempty ))
	:effect (and (clear ?y)
		(holding ?x)
		(not (clear ?x))
		(not (handempty ))
		(not (on ?x ?y))
		(not (ontable ?y)) 
		))

)