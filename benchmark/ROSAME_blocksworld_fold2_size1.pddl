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
	:precondition (and (ontable ?x) (clear ?x) (handempty) (holding ?x))
	:effect (and  (not (ontable ?x))  (not (clear ?x))  (not (handempty))  (not (holding ?x))))

(:action put_down
	:parameters (?x - block)
	:precondition (and (ontable ?x) (clear ?x))
	:effect (and  (not (ontable ?x))  (not (clear ?x))))

(:action stack
	:parameters (?x - block ?y - block)
	:precondition (and )
	:effect (and (on ?x ?y) (on ?y ?x) (ontable ?x) (ontable ?y) (clear ?x) (clear ?y) (handempty) (holding ?x) (holding ?y)))

(:action unstack
	:parameters (?x - block ?y - block)
	:precondition (and (on ?x ?y) (clear ?x) (handempty))
	:effect (and (clear ?y) (holding ?x) (not (on ?x ?y))  (not (clear ?x))  (not (handempty))))

)