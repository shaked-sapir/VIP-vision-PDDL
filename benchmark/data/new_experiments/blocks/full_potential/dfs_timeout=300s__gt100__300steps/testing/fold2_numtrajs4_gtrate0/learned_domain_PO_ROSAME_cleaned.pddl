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
	:effect (and (holding ?x) (not (ontable ?x))  (not (handempty))))

(:action put_down
	:parameters (?x - block)
	:precondition (and )
	:effect (and ))

(:action stack
	:parameters (?x - block ?y - block)
	:precondition (and (clear ?x) (holding ?x))
	:effect (and (on ?x ?y) (on ?y ?x) (handempty) (not (clear ?x))  (not (holding ?x))))

(:action unstack
	:parameters (?x - block ?y - block)
	:precondition (and )
	:effect (and (on ?x ?y) (ontable ?y) (clear ?x)))

)