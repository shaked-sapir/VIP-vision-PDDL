(define (domain maze)
(:requirements :strips :typing)
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
	:precondition (and (clear ?to) (at ?p ?from))
	:effect (and (move-dir-up ?from ?to) (move-dir-down ?to ?from) (clear ?from) (at ?p ?to) (oriented-up ?p) (not (clear ?to))  (not (at ?p ?from))))

(:action move_down
	:parameters (?p - player ?from - location ?to - location)
	:precondition (and (move-dir-up ?to ?from) (move-dir-down ?from ?to) (clear ?to) (at ?p ?from) (oriented-left ?p))
	:effect (and (clear ?from) (at ?p ?to) (oriented-down ?p) (not (clear ?to))  (not (at ?p ?from))  (not (oriented-left ?p))))

(:action move_left
	:parameters (?p - player ?from - location ?to - location)
	:precondition (and )
	:effect (and ))

(:action move_right
	:parameters (?p - player ?from - location ?to - location)
	:precondition (and (move-dir-left ?to ?from) (move-dir-right ?from ?to) (clear ?to) (at ?p ?from))
	:effect (and (clear ?from) (at ?p ?to) (oriented-right ?p) (not (clear ?to))  (not (at ?p ?from))))

)