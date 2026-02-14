(define (domain maze)
(:requirements :strips :equality :negative-preconditions :typing)
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

(:action move_down
	:parameters (?p - player ?from - location ?to - location)
	:precondition (and (oriented-down ?p))
	:effect (and  
		))

(:action move_left
	:parameters (?p - player ?from - location ?to - location)
	:precondition (and (at ?p ?from)
	(clear ?to))
	:effect (and (at ?p ?to)
		(clear ?from)
		(not (at ?p ?from))
		(not (clear ?to))
		(not (oriented-left ?p))
		(oriented-right ?p) 
		))

)