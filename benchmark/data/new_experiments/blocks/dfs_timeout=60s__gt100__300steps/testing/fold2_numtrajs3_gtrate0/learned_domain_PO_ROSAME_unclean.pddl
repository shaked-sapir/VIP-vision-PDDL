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
	:precondition (and )
	:effect (and (ontable ?x) (clear ?x) (handempty) (holding ?x)))

(:action stack
	:parameters (?x - block ?y - block)
	:precondition (and (ontable ?y) (clear ?y) (holding ?x))
	:effect (and (on ?x ?y) (clear ?x) (handempty) (not (ontable ?y))  (not (clear ?y))  (not (holding ?x))))

(:action unstack
	:parameters (?x - block ?y - block)
	:precondition (and (ontable ?y) (handempty) (holding ?x))
	:effect (and  (not (ontable ?y))  (not (handempty))  (not (holding ?x))))

)