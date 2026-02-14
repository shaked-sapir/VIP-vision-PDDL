(define (domain blocks)
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
	:precondition (and (clear ?x)
	(handempty )
	(ontable ?x))
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
	:effect (and (clear ?x)
		(handempty )
		(not (clear ?y))
		(not (holding ?x))
		(on ?x ?y)
		(ontable ?x) 
		))

(:action unstack
	:parameters (?x - block ?y - block)
	:precondition (and (handempty ))
	:effect (and (holding ?x)
		(not (clear ?x))
		(not (clear ?y))
		(not (handempty ))
		(not (on ?x ?y))
		(not (ontable ?x))
		(ontable ?y) 
		))

)