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
	:precondition (and (ontable ?x) (clear ?x))
	:effect (and (holding ?x) (not (ontable ?x))  (not (clear ?x))))

(:action put_down
	:parameters (?x - block)
	:precondition (and (ontable ?x) (clear ?x) (handempty) (holding ?x))
	:effect (and ))

(:action stack
	:parameters (?x - block ?y - block)
	:precondition (and (clear ?y) (holding ?x))
	:effect (and (on ?x ?y) (clear ?x) (not (clear ?y))  (not (holding ?x))))

(:action unstack
	:parameters (?x - block ?y - block)
	:precondition (and (on ?x ?y) (on ?y ?x) (ontable ?x) (ontable ?y) (clear ?x) (clear ?y) (handempty) (holding ?x) (holding ?y))
	:effect (and  (not (on ?x ?y))  (not (on ?y ?x))  (not (ontable ?x))  (not (ontable ?y))  (not (clear ?x))  (not (clear ?y))  (not (handempty))  (not (holding ?x))  (not (holding ?y))))

)