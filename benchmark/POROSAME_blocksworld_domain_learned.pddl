(define (domain blocksworld)
(:requirements :typing :strips)
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
	:precondition (and (handempty))
	:effect (and  (not (handempty))))

(:action put_down
	:parameters (?x - block)
	:precondition (and )
	:effect (and (handempty)))

(:action stack
	:parameters (?x - block ?y - block)
	:precondition (and )
	:effect (and (clear ?y) (handempty)))

(:action unstack
	:parameters (?x - block ?y - block)
	:precondition (and (handempty))
	:effect (and (clear ?y) (not (handempty))))

)