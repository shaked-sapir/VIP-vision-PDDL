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
	:precondition (and (holding ?x))
	:effect (and (ontable ?x) (clear ?x) (handempty) (not (holding ?x))))

(:action stack
	:parameters (?x - block ?y - block)
	:precondition (and )
	:effect (and (clear ?x) (handempty)))

(:action unstack
	:parameters (?x - block ?y - block)
	:precondition (and (clear ?x) (handempty))
	:effect (and (holding ?x) (not (clear ?x))  (not (handempty))))

)