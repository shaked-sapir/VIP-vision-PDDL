(define (domain maze)
(:requirements :equality :strips :typing :negative-preconditions)
(:types 	player location - object
)

(:predicates (move-dir-up ?v0 - location ?v1 - location)
	(move-dir-down ?v0 - location ?v1 - location)
	(move-dir-left ?v0 - location ?v1 - location)
	(move-dir-right ?v0 - location ?v1 - location)
	(clear ?v0 - location)
	(at ?v0 - player ?v1 - location)
	(oriented-up ?v0 - player)
	(oriented-down ?v0 - player)
	(oriented-left ?v0 - player)
	(oriented-right ?v0 - player)
	(is-goal ?v0 - location)
)

(:action move_up
	:parameters (?p - player ?from - location ?to - location)
	:precondition (and )
	:effect (and (not (oriented-right ?p))
		(oriented-up ?p) 
		))

(:action move_down
	:parameters (?p - player ?from - location ?to - location)
	:precondition (and (oriented-down ?p))
	:effect (and  
		))

(:action move_left
	:parameters (?p - player ?from - location ?to - location)
	:precondition (and (clear ?to))
	:effect (and (not (oriented-up ?p))
		(oriented-right ?p) 
		))

(:action move_right
	:parameters (?p - player ?from - location ?to - location)
	:precondition (and (clear ?to))
	:effect (and (not (oriented-down ?p))
		(oriented-up ?p) 
		))

)