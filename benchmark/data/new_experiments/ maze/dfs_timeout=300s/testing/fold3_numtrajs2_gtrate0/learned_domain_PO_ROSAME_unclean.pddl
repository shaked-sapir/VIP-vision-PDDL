(define (domain maze)
(:requirements :typing :strips)
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
	:effect (and ))

(:action move_down
	:parameters (?p - player ?from - location ?to - location)
	:precondition (and (oriented-up ?p))
	:effect (and (at ?p ?from) (oriented-right ?p) (not (oriented-up ?p))))

(:action move_left
	:parameters (?p - player ?from - location ?to - location)
	:precondition (and (at ?p ?to))
	:effect (and ))

(:action move_right
	:parameters (?p - player ?from - location ?to - location)
	:precondition (and (move-dir-left ?to ?from) (move-dir-right ?from ?to) (clear ?from) (clear ?to))
	:effect (and (at ?p ?from) (at ?p ?to) (oriented-up ?p) (oriented-right ?p) (is-goal ?to) (not (clear ?from))  (not (clear ?to))))

)