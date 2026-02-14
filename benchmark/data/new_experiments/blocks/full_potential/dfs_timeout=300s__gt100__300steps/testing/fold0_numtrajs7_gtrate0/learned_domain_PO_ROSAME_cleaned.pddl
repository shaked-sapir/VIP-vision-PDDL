(define (domain blocks)
(:requirements :strips :typing)
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
	:precondition (and (clear ?x) (handempty))
	:effect (and (holding ?x) (not (handempty))))

(:action put_down
	:parameters (?x - block)
	:precondition (and (clear ?x) (holding ?x))
	:effect (and (ontable ?x) (handempty) (not (holding ?x))))

(:action stack
	:parameters (?x - block ?y - block)
	:precondition (and (ontable ?y) (clear ?x) (clear ?y) (holding ?x))
	:effect (and (on ?x ?y) (on ?y ?x) (handempty) (not (ontable ?y))  (not (clear ?y))  (not (holding ?x))))

(:action unstack
	:parameters (?x - block ?y - block)
	:precondition (and (on ?x ?y) (clear ?x) (handempty))
	:effect (and (clear ?y) (holding ?x) (not (on ?x ?y))  (not (handempty))))

)