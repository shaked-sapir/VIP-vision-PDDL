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
	:precondition (and (move-dir-down ?to ?from) (clear ?to))
	:effect (and (move-dir-up ?from ?to) (clear ?from) (at ?p ?to) (oriented-up ?p) (not (move-dir-down ?to ?from))  (not (clear ?to))))

(:action move_down
	:parameters (?p - player ?from - location ?to - location)
	:precondition (and )
	:effect (and ))

(:action move_left
	:parameters (?p - player ?from - location ?to - location)
	:precondition (and (move-dir-left ?from ?to) (move-dir-right ?to ?from) (clear ?from) (clear ?to) (oriented-up ?p))
	:effect (and (oriented-right ?p) (not (oriented-up ?p))))

(:action move_right
	:parameters (?p - player ?from - location ?to - location)
	:precondition (and (oriented-down ?p))
	:effect (and (move-dir-left ?to ?from) (move-dir-right ?from ?to) (at ?p ?from) (oriented-right ?p) (not (oriented-down ?p))))

)