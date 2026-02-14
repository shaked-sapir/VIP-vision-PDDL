(define (domain blocks)
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
	:precondition (and (ontable ?x) (handempty))
	:effect (and (clear ?x) (holding ?x) (not (ontable ?x))  (not (handempty))))

(:action put_down
	:parameters (?x - block)
	:precondition (and (holding ?x))
	:effect (and (ontable ?x) (clear ?x) (handempty) (not (holding ?x))))

(:action stack
	:parameters (?x - block ?y - block)
	:precondition (and (clear ?x) (holding ?x))
	:effect (and (on ?x ?y) (on ?y ?x) (handempty) (not (clear ?x))  (not (holding ?x))))

(:action unstack
	:parameters (?x - block ?y - block)
	:precondition (and (on ?x ?y) (ontable ?x) (ontable ?y) (handempty))
	:effect (and (clear ?x) (clear ?y) (holding ?x) (not (on ?x ?y))  (not (ontable ?x))  (not (ontable ?y))  (not (handempty))))

)