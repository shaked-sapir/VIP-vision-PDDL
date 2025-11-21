(define (domain blocks)
(:requirements :strips :typing :equality :negative-preconditions)
(:types 	block robot - object
)

(:predicates (on ?x - block ?y - block)
	(ontable ?x - block)
	(clear ?x - block)
	(handempty ?x - robot)
	(handfull ?x - robot)
	(holding ?x - block)
)

(:action pick-up
	:parameters (?x - block ?robot - robot)
	:precondition (and (handempty ?robot)
	(not (handfull ?robot)))
	:effect (and (handfull ?robot)
		(holding ?x)
		(not (clear ?x))
		(not (handempty ?robot))
		(not (ontable ?x)) 
		))

(:action put-down
	:parameters (?x - block ?robot - robot)
	:precondition (and )
	:effect (and (clear ?x)
		(handempty ?robot)
		(not (handfull ?robot))
		(not (holding ?x)) 
		))

(:action stack
	:parameters (?x - block ?y - block ?robot - robot)
	:precondition (and (not (= ?x ?y)))
	:effect (and (clear ?x)
		(handempty ?robot)
		(not (clear ?y))
		(not (handfull ?robot))
		(not (holding ?x))
		(on ?x ?y) 
		))

(:action unstack
	:parameters (?x - block ?y - block ?robot - robot)
	:precondition (and (not (handfull ?robot))(not (= ?x ?y)))
	:effect (and (clear ?y)
		(handfull ?robot)
		(holding ?x)
		(not (clear ?x))
		(not (handempty ?robot))
		(not (on ?x ?y)) 
		))

)