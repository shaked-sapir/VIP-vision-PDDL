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
	:precondition (and (ontable ?x) (clear ?x) (handempty))
	:effect (and (holding ?x) (not (ontable ?x))  (not (clear ?x))  (not (handempty))))

(:action put_down
	:parameters (?x - block)
	:precondition (and (ontable ?x) (clear ?x) (handempty) (holding ?x))
	:effect (and ))

(:action stack
	:parameters (?x - block ?y - block)
	:precondition (and (ontable ?y) (clear ?y) (holding ?x))
	:effect (and (on ?x ?y) (clear ?x) (handempty) (not (ontable ?y))  (not (clear ?y))  (not (holding ?x))))

(:action unstack
	:parameters (?x - block ?y - block)
	:precondition (and (on ?x ?y) (ontable ?y) (clear ?x) (clear ?y) (holding ?x))
	:effect (and ))

)