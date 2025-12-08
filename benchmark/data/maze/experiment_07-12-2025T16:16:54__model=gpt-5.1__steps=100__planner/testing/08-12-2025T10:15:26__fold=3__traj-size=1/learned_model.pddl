(define (domain maze)
(:requirements :equality :negative-preconditions :strips :typing)
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

(:action move-up
	:parameters (?p - player ?from - location ?to - location)
	:precondition (and (at ?p ?from)
	(clear ?to)
	(move-dir-down ?to ?from)
	(move-dir-up ?from ?to)
	(oriented-left ?p))
	:effect (and (oriented-up ?p) 
		))

(:action move-down
	:parameters (?p - player ?from - location ?to - location)
	:precondition (and (at ?p ?from)
	(clear ?to)
	(move-dir-down ?from ?to)
	(move-dir-up ?to ?from))
	:effect (and (oriented-up ?p) 
		))

(:action move-left
	:parameters (?p - player ?from - location ?to - location)
	:precondition (and (at ?p ?from)
	(clear ?to)
	(move-dir-left ?from ?to)
	(move-dir-right ?to ?from)
	(oriented-right ?p))
	:effect (and (at ?p ?to)
		(clear ?from) 
		))

)