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
	:precondition (and (clear ?x)
	(handempty )
	(ontable ?x))
	:effect (and  
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
	:precondition (and (clear ?y)
	(holding ?x)
	(ontable ?y))
	:effect (and  
		))

(:action unstack
	:parameters (?x - block ?y - block)
	:precondition (and (handempty )
	(on ?x ?y))
	:effect (and (clear ?y)
		(holding ?x)
		(not (clear ?x))
		(not (handempty ))
		(not (on ?x ?y))
		(not (ontable ?x)) 
		))

)