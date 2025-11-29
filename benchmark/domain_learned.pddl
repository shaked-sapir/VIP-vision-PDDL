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
	:precondition (and (ontable ?x) (clear ?x) (handempty))
	:effect (and  (not (ontable ?x))  (not (clear ?x))  (not (handempty))))

(:action put_down
	:parameters (?x - block)
	:precondition (and (holding ?x))
	:effect (and  (not (holding ?x))))

(:action stack
	:parameters (?x - block ?y - block)
	:precondition (and (ontable ?y) (clear ?y) (holding ?x))
	:effect (and  (not (ontable ?y))  (not (clear ?y))  (not (holding ?x))))

(:action unstack
	:parameters (?x - block ?y - block)
	:precondition (and (on ?x ?y) (ontable ?y) (clear ?x) (handempty))
	:effect (and  (not (on ?x ?y))  (not (ontable ?y))  (not (clear ?x))  (not (handempty))))

)