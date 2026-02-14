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
	:precondition (and (ontable ?x) (clear ?x) (handempty))
	:effect (and (holding ?x) (not (ontable ?x))  (not (clear ?x))  (not (handempty))))

(:action put_down
	:parameters (?x - block)
	:precondition (and (holding ?x))
	:effect (and (ontable ?x) (clear ?x) (handempty)))

(:action stack
	:parameters (?x - block ?y - block)
	:precondition (and (ontable ?y) (holding ?x))
	:effect (and (on ?x ?y) (on ?y ?x) (ontable ?x) (clear ?y) (handempty) (not (ontable ?y))  (not (holding ?x))))

(:action unstack
	:parameters (?x - block ?y - block)
	:precondition (and (on ?x ?y) (handempty))
	:effect (and (holding ?x) (not (on ?x ?y))  (not (handempty))))

)